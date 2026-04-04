"""
v28_node_attn.py — ADIA Causal Discovery
  Node-centric 8ch 2D density + node self-attention (NO edge pipeline)

The cleanest architecture yet. No conv1d, no edge tensors, no O(p²) preprocessing.

Per node v:
  8ch 32×32 image → hierarchical conv2d → node_emb (d=64)
Cross-node:
  Stack K node embeddings → NodeSelfAttention (2 layers) → context-aware embeddings
Per node:
  Attended embedding → 8-class head → classification

Why this works:
  - The 8ch image (4 raw scatter + 4 ANM residual) captures v's full relationship
    with X and Y, including directional asymmetry
  - Node self-attention lets nodes see each other: "if node 3 looks like a confounder
    and node 5 looks like a mediator, that constrains the graph topology"
  - O(K²) attention where K=p-2 (typically ~8), vs O(p²(p-1)²) for edge attention
  - The 8-class label IS the adjacency matrix (each class = specific 4-bit edge pattern)

Channels:
  ch0: density(v, X)       — raw joint scatter
  ch1: density(v, Y)       — raw joint scatter
  ch2: density(X, v)       — transpose view
  ch3: density(Y, v)       — transpose view
  ch4: density(v, resid_X) — ANM residual of X, paired with v
  ch5: density(X, resid_v) — ANM residual of v, paired with X
  ch6: density(v, resid_Y) — ANM residual of Y, paired with v
  ch7: density(Y, resid_v) — ANM residual of v, paired with Y

Usage:
    python v28_node_attn.py
"""

# @crunch/keep:on
import crunch

import os

import typing
import gc
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.autograd.graph
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

import networkx as nx

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# Configuration
# ============================================================
N_OBS: int = 1000
N_KERNEL: int = 1000
GRID_SIZE: int = 32
N_NODE_CHANNELS: int = 8  # 4 raw + 4 ANM
SCATTER_SIGMA: float = 5.0
ANM_BW: float = 0.5
N_CLASSES: int = 8
D_MODEL: int = 64
N_HEADS: int = 4
N_ATTN_LAYERS: int = 2
CLASS_NAMES: list[str] = [
    "Confounder", "Collider", "Mediator",
    "Cause of X", "Cause of Y",
    "Consequence of X", "Consequence of Y",
    "Independent",
]

MAX_EPOCHS: int = 30
BATCH_SIZE: int = 16
LR: float = 1e-3
LOCAL_CACHE_DIR: str = "dataset_cache/"
IS_CLOUD_SUBMIT: bool = False


# ============================================================
# Graph Utilities (identical to v11)
# ============================================================
def graph_nodes_representation(graph, nodelist):
    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=nodelist).todense()
    return tuple(adjacency_matrix.flatten())


def create_graph_label():
    graph_label = {
        nx.DiGraph([("X", "Y"), ("v", "X"), ("v", "Y")]): "Confounder",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("Y", "v")]): "Collider",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("v", "Y")]): "Mediator",
        nx.DiGraph([("X", "Y"), ("v", "X")]): "Cause of X",
        nx.DiGraph([("X", "Y"), ("v", "Y")]): "Cause of Y",
        nx.DiGraph([("X", "Y"), ("X", "v")]): "Consequence of X",
        nx.DiGraph([("X", "Y"), ("Y", "v")]): "Consequence of Y",
        nx.DiGraph({"X": ["Y"], "v": []}): "Independent",
    }
    nodelist = ["v", "X", "Y"]
    adjacency_label = {
        graph_nodes_representation(graph, nodelist): label
        for graph, label in graph_label.items()
    }
    return graph_label, adjacency_label


_GRAPH_LABEL, _ADJACENCY_LABEL = None, None


def get_adjacency_label():
    global _GRAPH_LABEL, _ADJACENCY_LABEL
    if _ADJACENCY_LABEL is None:
        _GRAPH_LABEL, _ADJACENCY_LABEL = create_graph_label()
    return _ADJACENCY_LABEL


def get_labels(adjacency_matrix, adjacency_label):
    result = {}
    for variable in adjacency_matrix.columns.drop(["X", "Y"]):
        submatrix = adjacency_matrix.loc[
            [variable, "X", "Y"], [variable, "X", "Y"]
        ]
        key = tuple(submatrix.values.flatten())
        result[variable] = adjacency_label[key]
    return result


def transform_proba_to_DAG(nodes, pred):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edge("X", "Y")
    x_index, y_index = np.unravel_index(
        np.argsort(pred.ravel())[::-1], pred.shape
    )
    for i, j in zip(x_index, y_index):
        n1, n2 = nodes[i], nodes[j]
        if i == j:
            continue
        if {n1, n2} == {"X", "Y"}:
            continue
        if pred[i, j] > 0.5:
            G.add_edge(n1, n2)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(n1, n2)
    return nx.to_numpy_array(G)


# ============================================================
# Multivariate kernel regression (for ANM residuals)
# ============================================================
def compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=0.5):
    N, p = data.shape
    n_sub = len(sub_idx)
    data_sub = data[sub_idx]

    diff = data_sub[:, None, :] - data_sub[None, :, :]
    sq_dist = (diff ** 2).sum(axis=-1)
    W = np.exp(-sq_dist / (2 * bandwidth ** 2))

    all_coeffs = {}
    for j in range(p):
        other_cols = [k for k in range(p) if k != j]
        X_design = np.concatenate([np.ones((n_sub, 1)), data_sub[:, other_cols]], axis=1)
        y_target = data_sub[:, j]
        A_all = np.einsum('il,la,lb->iab', W, X_design, X_design)
        b_all = np.einsum('il,la,l->ia', W, X_design, y_target)
        reg = 1e-6 * np.eye(p)[None, :, :]
        try:
            c_all = np.linalg.solve(A_all + reg, b_all[:, :, None]).squeeze(-1)
        except np.linalg.LinAlgError:
            c_all = np.zeros((n_sub, p))
        all_coeffs[j] = (c_all, other_cols)

    dist_to_sub = np.sum((data[:, None, :] - data_sub[None, :, :]) ** 2, axis=-1)
    nearest = np.argmin(dist_to_sub, axis=1)

    coeff_map, resid_map = {}, {}
    for j in range(p):
        c_all, other_cols = all_coeffs[j]
        for idx_in_other, k in enumerate(other_cols):
            coeff_map[(k, j)] = c_all[:, idx_in_other + 1][nearest].astype(np.float32)
        c_nn = c_all[nearest]
        X_full = np.concatenate([np.ones((N, 1)), data[:, other_cols]], axis=1)
        y_hat = np.sum(c_nn * X_full, axis=1)
        resid_map[j] = (data[:, j] - y_hat).astype(np.float32)

    return coeff_map, resid_map


# ============================================================
# 2D Preprocessing — 8-channel node scatter density
# ============================================================
def build_scatter_density(source, target, grid_size=32, sigma=2.0):
    """Render scatter plot as density image. Min-max, joint density."""
    s_min, s_max = source.min(), source.max()
    t_min, t_max = target.min(), target.max()
    if s_max - s_min < 1e-10:
        s_norm = np.full_like(source, 0.5)
    else:
        s_norm = (source - s_min) / (s_max - s_min)
    if t_max - t_min < 1e-10:
        t_norm = np.full_like(target, 0.5)
    else:
        t_norm = (target - t_min) / (t_max - t_min)

    hist, _, _ = np.histogram2d(
        s_norm, t_norm, bins=grid_size, range=[[0, 1], [0, 1]]
    )
    hist_smooth = gaussian_filter(hist.astype(np.float32), sigma=sigma)
    total = hist_smooth.sum()
    if total > 0:
        hist_smooth /= total
    return hist_smooth


def build_node_images(df, resid_map=None):
    """Build 8-channel scatter density images for each non-X/Y node."""
    cols = list(df.columns)
    data = df.values.astype(np.float32)
    col2idx = {c: i for i, c in enumerate(cols)}
    xi = col2idx["X"]
    yi = col2idx["Y"]
    other_nodes = [c for c in cols if c not in ("X", "Y")]

    x_data = data[:, xi]
    y_data = data[:, yi]

    if resid_map is None:
        N = data.shape[0]
        n_sub = min(N_KERNEL, N)
        sub_idx = np.random.choice(N, n_sub, replace=False)
        _, resid_map = compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=ANM_BW)

    resid_X = resid_map[xi]
    resid_Y = resid_map[yi]

    node_images = {}
    for v_name in other_nodes:
        vi = col2idx[v_name]
        v_data = data[:, vi]
        resid_v = resid_map[vi]

        ch0 = build_scatter_density(v_data, x_data, GRID_SIZE, sigma=SCATTER_SIGMA)
        ch1 = build_scatter_density(v_data, y_data, GRID_SIZE, sigma=SCATTER_SIGMA)
        ch2 = build_scatter_density(x_data, v_data, GRID_SIZE, sigma=SCATTER_SIGMA)
        ch3 = build_scatter_density(y_data, v_data, GRID_SIZE, sigma=SCATTER_SIGMA)
        ch4 = build_scatter_density(v_data, resid_X, GRID_SIZE, sigma=SCATTER_SIGMA)
        ch5 = build_scatter_density(x_data, resid_v, GRID_SIZE, sigma=SCATTER_SIGMA)
        ch6 = build_scatter_density(v_data, resid_Y, GRID_SIZE, sigma=SCATTER_SIGMA)
        ch7 = build_scatter_density(y_data, resid_v, GRID_SIZE, sigma=SCATTER_SIGMA)

        node_images[v_name] = np.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7], axis=0)

    return node_images


# ============================================================
# Sample Builder
# ============================================================
def _build_single(args):
    df, y_df = args
    data = df.values.astype(np.float32)
    N = data.shape[0]
    n_sub = min(N_KERNEL, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)

    _, resid_map = compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=ANM_BW)
    node_images = build_node_images(df, resid_map=resid_map)

    cols = list(df.columns)
    p = len(cols)
    other_nodes = [c for c in cols if c not in ("X", "Y")]

    result = {
        "cols": cols,
        "p": p,
        "node_images": node_images,
        "other_nodes": other_nodes,
    }

    if y_df is not None:
        adj_np = y_df.values.astype(np.float32)
        adj_cols = list(y_df.columns)

        _, adjacency_label = create_graph_label()
        df_adj = pd.DataFrame(adj_np, columns=adj_cols, index=adj_cols)
        node_labels_dict = get_labels(df_adj, adjacency_label)
        node_labels = [CLASS_NAMES.index(node_labels_dict[n]) for n in other_nodes]
        result["node_labels"] = np.array(node_labels, dtype=np.int64)

    return result


# ============================================================
# Dataset & Collate
# ============================================================
class InMemoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        result = {
            "cols": item["cols"],
            "p": item["p"],
            "other_nodes": item["other_nodes"],
            "node_images": {k: torch.from_numpy(v) for k, v in item["node_images"].items()},
        }
        if "node_labels" in item:
            result["node_labels"] = torch.from_numpy(item["node_labels"])
        return result


def collate_fn(batch):
    B = len(batch)
    max_K = max(len(item["other_nodes"]) for item in batch)

    node_imgs = torch.zeros(B, max_K, N_NODE_CHANNELS, GRID_SIZE, GRID_SIZE)
    node_mask = torch.zeros(B, max_K, dtype=torch.bool)
    node_labels = torch.full((B, max_K), -1, dtype=torch.long)

    cols_list = []
    has_labels = False

    for b, item in enumerate(batch):
        cols_list.append(item["cols"])
        other = item["other_nodes"]
        for k, v_name in enumerate(other):
            node_imgs[b, k] = item["node_images"][v_name]
            node_mask[b, k] = True

        if "node_labels" in item:
            has_labels = True
            K = len(other)
            node_labels[b, :K] = item["node_labels"]

    out = {
        "node_imgs": node_imgs,
        "node_mask": node_mask,
        "cols": cols_list,
    }
    if has_labels:
        out["node_labels"] = node_labels
    return out


# ============================================================
# Model Architecture
# ============================================================
class NodeFeatureExtractor2D(nn.Module):
    """Hierarchical conv2d. 32×32 → 16×16 → 8×8 → 4×4 → pool → d"""
    def __init__(self, d=64, n_channels=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, d),
        )

    def forward(self, x):
        return self.net(x)


class NodeSelfAttention(nn.Module):
    """Self-attention across nodes within a graph.

    Lets nodes see each other: "if node 3 looks like a confounder
    and node 5 looks like a mediator, that constrains the topology."

    O(K²) where K = p-2 (typically ~8 nodes). Lightweight.
    """
    def __init__(self, d=64, n_heads=4):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.norm2 = nn.LayerNorm(d)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, K, d) — K node embeddings
        key_padding_mask: (B, K) — True for padded positions
        """
        B, K, _ = x.shape
        Q = self.q_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        K_ = self.k_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K_.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask_expanded, float('-inf'))

        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, K, self.d)
        out = self.out_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class ADIAModel(nn.Module):
    """v28: Node-centric 8ch 2D + node self-attention.

    No edge pipeline. No conv1d. No O(p²) preprocessing.
    Just: conv2d per node → self-attention across nodes → classify.
    """

    def __init__(self, d=None, aug_noise_std=0.0):
        super().__init__()
        d = d or D_MODEL
        self.d = d

        # Per-node conv2d feature extraction
        self.node_extractor = NodeFeatureExtractor2D(d, n_channels=N_NODE_CHANNELS)

        # Cross-node self-attention
        self.attn_layers = nn.ModuleList(
            [NodeSelfAttention(d, n_heads=N_HEADS) for _ in range(N_ATTN_LAYERS)]
        )

        # Per-node classification
        self.node_head = nn.Linear(d, N_CLASSES)

    def forward(self, node_imgs, node_mask=None, **kwargs):
        """
        node_imgs: (B, K, 8, 32, 32)
        node_mask: (B, K) bool — True for valid nodes
        Returns: list of (K_b, 8) tensors
        """
        B, K, C, H, W = node_imgs.shape

        # Extract per-node features via conv2d
        flat_imgs = node_imgs.view(B * K, C, H, W)
        flat_emb = self.node_extractor(flat_imgs)  # (B*K, d)
        node_embs = flat_emb.view(B, K, self.d)    # (B, K, d)

        # Cross-node self-attention
        inv_mask = ~node_mask if node_mask is not None else None
        for layer in self.attn_layers:
            node_embs = layer(node_embs, key_padding_mask=inv_mask)

        # Per-node classification
        all_logits = self.node_head(node_embs)  # (B, K, N_CLASSES)

        node_logits_list = []
        for b in range(B):
            K_b = int(node_mask[b].sum().item()) if node_mask is not None else K
            if K_b > 0:
                node_logits_list.append(all_logits[b, :K_b])
            else:
                node_logits_list.append(None)

        return node_logits_list


# ============================================================
# Training Wrapper
# ============================================================
def compute_class_weights(y_list):
    adjacency_label = get_adjacency_label()
    counts = torch.zeros(N_CLASSES)
    for y_df in y_list:
        cols = list(y_df.columns)
        arr = y_df.values
        col_idx = {c: i for i, c in enumerate(cols)}
        for v in cols:
            if v in ("X", "Y"):
                continue
            idx = [col_idx[v], col_idx["X"], col_idx["Y"]]
            sub = arr[np.ix_(idx, idx)]
            key = tuple(sub.flatten())
            counts[CLASS_NAMES.index(adjacency_label.get(key, "Independent"))] += 1
    w = 1.0 / (counts + 1e-6)
    return w / w.sum() * N_CLASSES


class ADIAModelWrapper(pl.LightningModule):
    def __init__(self, d=None, node_class_weights=None,
                 lr=1e-3, max_epochs=20):
        super().__init__()
        d = d or D_MODEL
        self.model = ADIAModel(d)
        self.lr = lr
        self.max_epochs = max_epochs
        self.node_criterion = nn.CrossEntropyLoss(
            weight=node_class_weights if node_class_weights is not None else torch.ones(N_CLASSES),
            ignore_index=-1,
        )

    def forward(self, batch):
        return self.model(
            node_imgs=batch["node_imgs"].to(self.device),
            node_mask=batch["node_mask"].to(self.device),
        )

    def _compute_loss(self, batch, split):
        node_logits_list = self(batch)
        B = batch["node_imgs"].shape[0]
        total_loss = torch.tensor(0.0, device=self.device)

        if "node_labels" in batch:
            all_nl, all_ll = [], []
            nl = batch["node_labels"].to(self.device)
            for b in range(B):
                if node_logits_list[b] is not None:
                    K = node_logits_list[b].shape[0]
                    all_ll.append(node_logits_list[b])
                    all_nl.append(nl[b, :K])
            if all_ll:
                cat_l = torch.cat(all_ll, dim=0)
                cat_n = torch.cat(all_nl, dim=0)
                valid = cat_n >= 0
                if valid.any():
                    total_loss = self.node_criterion(cat_l[valid], cat_n[valid])
                    self.log(f"{split}/node_loss", total_loss, prog_bar=True, sync_dist=True)

        self.log(f"{split}/loss", total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]


# ============================================================
# Train & Infer
# ============================================================
def train(X_train, y_train, model_directory_path):
    keys = list(X_train.keys())
    X_list = [X_train[k] for k in keys]
    y_list = [y_train[k] for k in keys]
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    node_w_path = os.path.join(LOCAL_CACHE_DIR, "node_weights_v8b.pt")
    if os.path.exists(node_w_path):
        node_w = torch.load(node_w_path, weights_only=True)
    else:
        node_w = compute_class_weights(y_list)
        torch.save(node_w, node_w_path)

    cache_path = os.path.join(LOCAL_CACHE_DIR, f"train_dataset_v28_nodeattn_nk{N_KERNEL}.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, "rb") as f:
            samples = pickle.load(f)
    else:
        args = [(X_list[i], y_list[i]) for i in range(len(X_list))]
        import multiprocessing as mp
        n_workers = max(1, mp.cpu_count() - 1)
        print(f"Building dataset ({len(args)} samples, {n_workers} workers)...")
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            raw = [None] * len(args)
            futures = {pool.submit(_build_single, a): idx for idx, a in enumerate(args)}
            for fut in tqdm(as_completed(futures), total=len(args)):
                raw[futures[fut]] = fut.result()
        samples = raw
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    dataset = InMemoryDataset(samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, collate_fn=collate_fn, pin_memory=True)
    print(f"Dataset: {len(dataset)} samples")

    wrapper = ADIAModelWrapper(d=D_MODEL, node_class_weights=node_w,
                                lr=LR, max_epochs=MAX_EPOCHS)
    n_params = sum(p.numel() for p in wrapper.model.parameters())
    print(f"Params: {n_params:,}")

    wandb_logger = WandbLogger(
        project="causal-discovery-thesis",
        name="v28_node_attn",
        config={
            "version": "v28",
            "grid_size": GRID_SIZE,
            "scatter_sigma": SCATTER_SIGMA,
            "anm_bw": ANM_BW,
            "n_node_channels": N_NODE_CHANNELS,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_attn_layers": N_ATTN_LAYERS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_epochs": MAX_EPOCHS,
            "n_params": n_params,
            "n_samples": len(dataset),
            "edge_pipeline": False,
            "node_self_attention": True,
        },
    )

    trainer = pl.Trainer(
        accelerator="gpu", devices=2,
        strategy="ddp",
        max_epochs=MAX_EPOCHS, precision="32-true",
        use_distributed_sampler=True,
        logger=wandb_logger, enable_checkpointing=True, enable_progress_bar=True,
    )
    trainer.fit(wrapper, loader)

    path = os.path.join(model_directory_path, "model_v28.pt")
    sd = wrapper.model.state_dict()
    torch.save({k.replace("module.", ""): v for k, v in sd.items()}, path)
    print(f"Model saved to {path}")


@torch.no_grad()
def infer_batch_local(dfs, model, device="cpu", batch_size=32, cache_dir=None):
    model = model.eval()
    cache_path = os.path.join(cache_dir, f"infer_v28_nk{N_KERNEL}.pkl") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            all_items = pickle.load(f)
    else:
        import multiprocessing as mp
        n_workers = max(1, mp.cpu_count() - 1)
        args = [(df, None) for df in dfs]
        print(f"Building node images infer ({len(dfs)} graphs, {n_workers} workers)...")
        ctx = mp.get_context('fork')
        all_items = [None] * len(args)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_build_single, a): idx
                for idx, a in enumerate(args)
            }
            for fut in tqdm(as_completed(futures), total=len(args)):
                all_items[futures[fut]] = fut.result()
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(all_items, f, protocol=pickle.HIGHEST_PROTOCOL)

    loader = DataLoader(InMemoryDataset(all_items), batch_size=batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_fn)
    results = [None] * len(all_items)
    patterns = {
        "Confounder": lambda n: [(n, "X"), (n, "Y")],
        "Collider": lambda n: [("X", n), ("Y", n)],
        "Mediator": lambda n: [("X", n), (n, "Y")],
        "Cause of X": lambda n: [(n, "X")],
        "Cause of Y": lambda n: [(n, "Y")],
        "Consequence of X": lambda n: [("X", n)],
        "Consequence of Y": lambda n: [("Y", n)],
        "Independent": lambda n: [],
    }
    idx_off = 0
    for batch in tqdm(loader, desc="inferring"):
        node_logits_list = model(
            node_imgs=batch["node_imgs"].to(device),
            node_mask=batch["node_mask"].to(device))
        B_cur = batch["node_imgs"].shape[0]
        for b in range(B_cur):
            item = all_items[idx_off + b]
            cols, p = item["cols"], item["p"]
            adj = np.zeros((p, p), dtype=int)
            col2idx = {c: i for i, c in enumerate(cols)}
            adj[col2idx["X"], col2idx["Y"]] = 1
            other_nodes = [n for n in cols if n not in ("X", "Y")]
            if node_logits_list[b] is not None:
                node_preds = torch.argmax(node_logits_list[b], dim=-1)
                for k, nn_ in enumerate(other_nodes):
                    for (s, d) in patterns[CLASS_NAMES[node_preds[k].item()]](nn_):
                        adj[col2idx[s], col2idx[d]] = 1
            A = pd.DataFrame(adj, columns=cols, index=cols)
            results[idx_off + b] = A
        idx_off += B_cur
    return results


def infer(X_test, model_directory_path, id_column_name, prediction_column_name):
    path = os.path.join(model_directory_path, "model_v28.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    dfs = [X_test[n] for n in X_test]
    cache_dir = None if IS_CLOUD_SUBMIT else LOCAL_CACHE_DIR
    adj_list = infer_batch_local(dfs, model, device=device, batch_size=64, cache_dir=cache_dir)
    submission = {}
    for name, A in zip(X_test.keys(), adj_list):
        for i in A.columns:
            for j in A.columns:
                submission[f"{name}_{i}_{j}"] = int(A.loc[i, j])
    s = pd.Series(submission).reset_index()
    s.columns = [id_column_name, prediction_column_name]
    return s


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("v28: Node-Centric 8ch 2D + Node Self-Attention")
    print(f"  Node: {N_NODE_CHANNELS}ch x {GRID_SIZE}x{GRID_SIZE} (sigma={SCATTER_SIGMA})")
    print(f"  Channels: 4 raw scatter + 4 ANM residual scatter")
    print(f"  Self-attention: {N_ATTN_LAYERS} layers, {N_HEADS} heads across K nodes")
    print(f"  NO edge pipeline, NO conv1d, NO O(p²) preprocessing")
    print(f"  Config: bs={BATCH_SIZE}, lr={LR}, epochs={MAX_EPOCHS}")
    print("=" * 60)

    X_train = pd.read_pickle("data/X_train.pickle")
    y_train = pd.read_pickle("data/y_train.pickle")
    print(f"Loaded {len(X_train)} training samples.")
    train(X_train, y_train, model_directory_path="resources")

    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank != 0:
        exit(0)

    X_test = pd.read_pickle("data/X_test_reduced.pickle")
    y_test = pd.read_pickle("data/y_test_reduced.pickle")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL)
    model.load_state_dict(torch.load("resources/model_v28.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    dfs = [X_test[n] for n in X_test]
    names = list(X_test.keys())
    adj_list = infer_batch_local(dfs, model, device=device, batch_size=64, cache_dir=LOCAL_CACHE_DIR)
    adjacency_label = get_adjacency_label()
    cc = {c: 0 for c in CLASS_NAMES}
    ct = {c: 0 for c in CLASS_NAMES}
    for name, A in zip(names, adj_list):
        pl_ = get_labels(A, adjacency_label)
        tl = get_labels(y_test[name], adjacency_label)
        for v in tl:
            ct[tl[v]] += 1
            if pl_.get(v, "Independent") == tl[v]:
                cc[tl[v]] += 1
    print("\nv28 Per-class accuracy:")
    accs = []
    for cls in CLASS_NAMES:
        n = ct[cls]
        acc = cc[cls] / n if n > 0 else 0.0
        accs.append(acc)
        print(f"  {cls:25s}: {acc:.4f}  (n={n})")
    print(f"\nv28 Balanced Accuracy: {np.mean(accs):.4f}")