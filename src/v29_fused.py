"""
v29_fused.py — ADIA Causal Discovery
  Edge-centric 2D: conv2d on per-edge scatter density images + structural attention.
  NO edge head, NO edge loss. Node classification only.

Based on v27 architecture but with fixes:
  - Raw [-1,1] histogram range (no min-max normalization)
  - No density normalization (no /=total)
  - sigma=5.0 (larger smoothing)
  - NO edge head, NO edge loss — edge pipeline is purely feature extraction
  - Node classification only via node_merge(4 edge embeddings)

Per directed edge (i->j), 4-channel 32x32 image:
  ch0: density(i, j)       — raw scatter
  ch1: density(j, i)       — transpose view
  ch2: density(i, resid_j) — ANM residual: if i->j true, this is uniform
  ch3: density(j, resid_i) — ANM residual: if i->j true, this has structure

Pipeline:
  edge image -> conv2d -> edge_emb + edge_type_emb -> merge
  -> structural self-attention x2 -> attended edge embeddings
  -> per node: node_merge(v->X, v->Y, X->v, Y->v) -> node_head -> 8-class

Usage:
    python v29_fused.py
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
from pytorch_lightning.strategies import DDPStrategy
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
ANM_BWS: list[float] = [0.2, 0.5, 1.0]
N_EDGE_2D_CHANNELS: int = 2 + 2 * len(ANM_BWS)  # 2 raw + 2*3 ANM = 8
SCATTER_SIGMA: float = 5.0
N_CLASSES: int = 8
N_EDGE_TYPES: int = 7
D_MODEL: int = 64
N_HEADS: int = 4
N_STRUCT_TYPES: int = 6
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
# Graph Utilities
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


# ============================================================
# Edge type
# ============================================================
def _edge_type(u_name, v_name):
    uX, uY = u_name == "X", u_name == "Y"
    vX, vY = v_name == "X", v_name == "Y"
    if uX and not vY:  return 0
    if uX and vY:      return 1
    if uY and not vX:  return 2
    if uY and vX:      return 3
    if not uX and not uY and vX: return 4
    if not uX and not uY and vY: return 5
    return 6


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
# 2D Preprocessing — raw [-1,1], no normalization
# ============================================================
def build_scatter_density(source, target, grid_size=32, sigma=5.0):
    """Raw [-1,1] range, Gaussian smoothed, NO density normalization."""
    hist, _, _ = np.histogram2d(
        source, target, bins=grid_size, range=[[-1, 1], [-1, 1]]
    )
    hist_smooth = gaussian_filter(hist.astype(np.float32), sigma=sigma)
    return hist_smooth


def build_edge_tensor_2d(df):
    """Build 8ch 2D scatter density for each directed edge (i->j).

    ch0: density(i, j)              — raw scatter
    ch1: density(j, i)              — transpose
    ch2: density(i, resid_j_bw0.2)  — ANM bw=0.2
    ch3: density(j, resid_i_bw0.2)  — ANM bw=0.2
    ch4: density(i, resid_j_bw0.5)  — ANM bw=0.5
    ch5: density(j, resid_i_bw0.5)  — ANM bw=0.5
    ch6: density(i, resid_j_bw1.0)  — ANM bw=1.0
    ch7: density(j, resid_i_bw1.0)  — ANM bw=1.0
    """
    cols = list(df.columns)
    p = len(cols)
    data = df.values.astype(np.float32)
    N = data.shape[0]

    n_sub = min(N_KERNEL, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)

    # Compute residuals at multiple bandwidths
    resid_maps = []
    for bw in ANM_BWS:
        _, rm = compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=bw)
        resid_maps.append(rm)

    edges = []
    edge_types = []

    for i, u_name in enumerate(cols):
        u_data = data[:, i]
        for j, v_name in enumerate(cols):
            if i == j:
                continue
            v_data = data[:, j]

            channels = [
                build_scatter_density(u_data, v_data, GRID_SIZE, SCATTER_SIGMA),  # raw
                build_scatter_density(v_data, u_data, GRID_SIZE, SCATTER_SIGMA),  # transpose
            ]
            for rm in resid_maps:
                channels.append(build_scatter_density(u_data, rm[j], GRID_SIZE, SCATTER_SIGMA))  # i vs resid_j
                channels.append(build_scatter_density(v_data, rm[i], GRID_SIZE, SCATTER_SIGMA))  # j vs resid_i

            edges.append(np.stack(channels, axis=0).astype(np.float32))
            edge_types.append(_edge_type(u_name, v_name))

    return np.stack(edges, axis=0), np.array(edge_types, dtype=np.int64)


# ============================================================
# Structural Relationship Matrix
# ============================================================
def build_struct_rel_matrix(p):
    E = p * (p - 1)
    edge_uv = []
    for u in range(p):
        for v in range(p):
            if u != v:
                edge_uv.append((u, v))
    rel = np.full((E, E), 5, dtype=np.int64)
    for e1 in range(E):
        u1, v1 = edge_uv[e1]
        for e2 in range(E):
            u2, v2 = edge_uv[e2]
            if u1 == v2 and v1 == u2:      rel[e1, e2] = 0
            elif u1 == u2:                  rel[e1, e2] = 1
            elif v1 == v2:                  rel[e1, e2] = 2
            elif v1 == u2:                  rel[e1, e2] = 3
            elif u1 == v2:                  rel[e1, e2] = 4
    return rel


_STRUCT_REL_CACHE = {}


def get_struct_rel_matrix(p):
    if p not in _STRUCT_REL_CACHE:
        _STRUCT_REL_CACHE[p] = build_struct_rel_matrix(p)
    return _STRUCT_REL_CACHE[p]


# ============================================================
# Sample Builder
# ============================================================
def _build_single(args):
    df, y_df = args
    edge_data, edge_types = build_edge_tensor_2d(df)

    cols = list(df.columns)
    p = len(cols)
    other_nodes = [c for c in cols if c not in ("X", "Y")]

    result = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "cols": cols,
        "p": p,
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
            "edge_data": torch.from_numpy(item["edge_data"]),
            "edge_types": torch.from_numpy(item["edge_types"]),
            "cols": item["cols"],
            "p": item["p"],
            "other_nodes": item["other_nodes"],
        }
        if "node_labels" in item:
            result["node_labels"] = torch.from_numpy(item["node_labels"])
        return result


def collate_fn(batch):
    max_E = max(item["edge_data"].shape[0] for item in batch)
    B = len(batch)

    edge_data = torch.zeros(B, max_E, N_EDGE_2D_CHANNELS, GRID_SIZE, GRID_SIZE)
    edge_types = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask = torch.zeros(B, max_E, dtype=torch.bool)
    struct_rel = torch.full((B, max_E, max_E), 5, dtype=torch.long)

    max_K = max(len(item["other_nodes"]) for item in batch)
    node_labels = torch.full((B, max_K), -1, dtype=torch.long)

    cols_list = []
    has_labels = False

    for b, item in enumerate(batch):
        E = item["edge_data"].shape[0]
        edge_data[b, :E] = item["edge_data"]
        edge_types[b, :E] = item["edge_types"]
        edge_mask[b, :E] = True
        cols_list.append(item["cols"])

        p = item["p"]
        rel_mat = get_struct_rel_matrix(p)
        struct_rel[b, :E, :E] = torch.from_numpy(rel_mat)

        if "node_labels" in item:
            has_labels = True
            K = len(item["other_nodes"])
            node_labels[b, :K] = item["node_labels"]

    out = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "edge_mask": edge_mask,
        "struct_rel": struct_rel,
        "cols": cols_list,
    }
    if has_labels:
        out["node_labels"] = node_labels
    return out


# ============================================================
# Model
# ============================================================
class MergeOperator(nn.Module):
    def __init__(self, n_inputs, d):
        super().__init__()
        self.linear = nn.Linear(n_inputs * d, d)
        self.norm = nn.LayerNorm(d)
        self.act = nn.GELU()

    def forward(self, inputs):
        return self.act(self.norm(self.linear(torch.cat(inputs, dim=-1))))


class EdgeFeatureExtractor2D(nn.Module):
    """Hierarchical conv2d: 32x32 -> 16x16 -> 8x8 -> 4x4 -> pool -> d"""
    def __init__(self, d=64, n_channels=4):
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


class StructuralSelfAttention(nn.Module):
    def __init__(self, d=64, n_heads=4, n_struct_types=6):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.struct_bias = nn.Embedding(n_struct_types, n_heads)
        nn.init.zeros_(self.struct_bias.weight)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, struct_rel, key_padding_mask=None):
        B, E, _ = x.shape
        Q = self.q_proj(x).view(B, E, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, E, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, E, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = attn + self.struct_bias(struct_rel).permute(0, 3, 1, 2)
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, E, self.d)
        out = self.out_proj(out)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class ADIAModel(nn.Module):
    """v29: Edge-centric 2D conv2d + structural attention.
    NO edge head. Node classification only via 4-edge merge."""

    def __init__(self, d=None):
        super().__init__()
        d = d or D_MODEL
        self.d = d

        # Edge pipeline — conv2d feature extractor
        self.extractor = EdgeFeatureExtractor2D(d, n_channels=N_EDGE_2D_CHANNELS)
        self.edge_type_emb = nn.Embedding(N_EDGE_TYPES, d)
        self.edge_merge = MergeOperator(n_inputs=2, d=d)
        self.attn_layers = nn.ModuleList(
            [StructuralSelfAttention(d, n_heads=N_HEADS, n_struct_types=N_STRUCT_TYPES)
             for _ in range(2)]
        )
        # NO edge_head

        # Node head — merge 4 edge embeddings -> 8-class
        self.node_merge = MergeOperator(n_inputs=4, d=d)
        self.node_head = nn.Linear(d, N_CLASSES)

    def forward(self, edge_data, edge_types, edge_mask, cols_list, struct_rel=None):
        B, E, C, H, W = edge_data.shape

        # Conv2d on edge images
        x_flat = edge_data.view(B * E, C, H, W)
        conv_emb = self.extractor(x_flat).view(B, E, self.d)

        type_emb = self.edge_type_emb(edge_types)
        edge_emb = self.edge_merge([conv_emb, type_emb])

        inv_mask = ~edge_mask
        if struct_rel is None:
            struct_rel = torch.full((B, E, E), 5, dtype=torch.long, device=edge_data.device)
            for b in range(B):
                p = len(cols_list[b])
                e = p * (p - 1)
                rel = get_struct_rel_matrix(p)
                struct_rel[b, :e, :e] = torch.from_numpy(rel).to(edge_data.device)

        for layer in self.attn_layers:
            edge_emb = layer(edge_emb, struct_rel, key_padding_mask=inv_mask)

        # Node classification — merge 4 edge embeddings per node
        node_logits_list = []
        for b in range(B):
            cols = cols_list[b]
            p = len(cols)
            col2idx = {c: i for i, c in enumerate(cols)}
            other = [c for c in cols if c not in ("X", "Y")]

            if not other:
                node_logits_list.append(None)
                continue

            xi, yi = col2idx["X"], col2idx["Y"]

            def _eidx(u, v):
                return u * (p - 1) + v - (1 if v > u else 0)

            node_logits = []
            for v_name in other:
                vi = col2idx[v_name]
                embs = [
                    edge_emb[b, _eidx(vi, xi)],  # v -> X
                    edge_emb[b, _eidx(vi, yi)],  # v -> Y
                    edge_emb[b, _eidx(xi, vi)],  # X -> v
                    edge_emb[b, _eidx(yi, vi)],  # Y -> v
                ]
                merged = self.node_merge(embs)
                node_logits.append(self.node_head(merged))

            node_logits_list.append(torch.stack(node_logits) if node_logits else None)

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
            batch["edge_data"].to(self.device),
            batch["edge_types"].to(self.device),
            batch["edge_mask"].to(self.device),
            batch["cols"],
            struct_rel=batch["struct_rel"].to(self.device),
        )

    def _compute_loss(self, batch, split):
        node_logits_list = self(batch)
        B = len(node_logits_list)
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

    cache_path = os.path.join(LOCAL_CACHE_DIR, f"train_dataset_v29_edge2d_nk{N_KERNEL}.pkl")
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
        name="v29_fused",
        config={
            "version": "v29",
            "scatter_sigma": SCATTER_SIGMA,
            "scatter_range": "raw [-1,1]",
            "anm_bws": ANM_BWS,
            "n_edge_2d_channels": N_EDGE_2D_CHANNELS,
            "d_model": D_MODEL,
            "edge_head": False,
            "edge_loss": False,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_epochs": MAX_EPOCHS,
            "n_params": n_params,
        },
    )

    trainer = pl.Trainer(
        accelerator="gpu", devices=2,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=MAX_EPOCHS, precision="32-true",
        use_distributed_sampler=True,
        logger=wandb_logger, enable_checkpointing=True, enable_progress_bar=True,
    )
    trainer.fit(wrapper, loader)

    path = os.path.join(model_directory_path, "model_v29.pt")
    sd = wrapper.model.state_dict()
    torch.save({k.replace("module.", ""): v for k, v in sd.items()}, path)
    print(f"Model saved to {path}")


@torch.no_grad()
def infer_batch_local(dfs, model, device="cpu", batch_size=32, cache_dir=None):
    model = model.eval()
    cache_path = os.path.join(cache_dir, f"infer_v29_nk{N_KERNEL}.pkl") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            all_items = pickle.load(f)
    else:
        import multiprocessing as mp
        n_workers = max(1, mp.cpu_count() - 1)
        args = [(df, None) for df in dfs]
        print(f"Building tensors infer ({len(dfs)} graphs, {n_workers} workers)...")
        ctx = mp.get_context('fork')
        all_items = [None] * len(args)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_build_single, a): idx for idx, a in enumerate(args)}
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
            batch["edge_data"].to(device), batch["edge_types"].to(device),
            batch["edge_mask"].to(device), batch["cols"],
            struct_rel=batch["struct_rel"].to(device))
        B_cur = len(node_logits_list)
        for b in range(B_cur):
            item = all_items[idx_off + b]
            cols, p = item["cols"], item["p"]
            adj = np.zeros((p, p), dtype=int)
            col2idx = {c: i for i, c in enumerate(cols)}
            adj[col2idx["X"], col2idx["Y"]] = 1
            other_nodes = item["other_nodes"]
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
    path = os.path.join(model_directory_path, "model_v29.pt")
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
    print("v29: Edge-Centric 2D Conv2d + Structural Attention")
    print(f"  Edge: {N_EDGE_2D_CHANNELS}ch {GRID_SIZE}x{GRID_SIZE} scatter (sigma={SCATTER_SIGMA}, raw [-1,1])")
    print(f"  Channels: 2 raw + 2x{len(ANM_BWS)} ANM (bw={ANM_BWS})")
    print(f"  Structural self-attention x2 (NO edge head, NO edge loss)")
    print(f"  Node: merge(v->X, v->Y, X->v, Y->v) -> 8-class")
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
    model.load_state_dict(torch.load("resources/model_v29.pt", map_location=device, weights_only=True))
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
    print("\nv29 Per-class accuracy:")
    accs = []
    for cls in CLASS_NAMES:
        n = ct[cls]
        acc = cc[cls] / n if n > 0 else 0.0
        accs.append(acc)
        print(f"  {cls:25s}: {acc:.4f}  (n={n})")
    print(f"\nv29 Balanced Accuracy: {np.mean(accs):.4f}")