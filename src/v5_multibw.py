"""
v5_multibw.py — v2 baseline + multi-scale bandwidth kernel regression

Changes from v2:
  - Config: BANDWIDTHS = [0.2, 0.5, 1.0], N_CHANNELS = 2 + len(BANDWIDTHS) = 5
  - build_edge_tensor: computes kernel regression at each bandwidth,
    producing 5 channels: [sorted_u, sorted_v, coeff_bw0.2, coeff_bw0.5, coeff_bw1.0]
  - StemLayer: Linear(5, d) instead of Linear(3, d)
  - Everything else unchanged from v2.

Usage:
    python v5_multibw.py
"""

import typing
import os
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pytorch_lightning as pl

import networkx as nx
from sklearn.model_selection import train_test_split

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# Configuration
# ============================================================
N_OBS = 1000
N_KERNEL = 1000
N_CLASSES = 8
N_EDGE_TYPES = 7
D_MODEL = 64
CLASS_NAMES = [
    "Confounder", "Collider", "Mediator",
    "Cause of X", "Cause of Y",
    "Consequence of X", "Consequence of Y",
    "Independent",
]

# === Multi-bandwidth config ===
BANDWIDTHS = [0.2, 0.5, 1.0]   # small, medium, large scale
N_CHANNELS = 2 + len(BANDWIDTHS)  # sorted_u + sorted_v + one coeff per bandwidth

MAX_EPOCHS = 30
LOCAL_CACHE_DIR = "dataset_cache/"


# ============================================================
# Graph Utilities (unchanged from v2)
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
# Data Preprocessing — Multi-bandwidth kernel regression
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


def compute_multivariate_kernel_coefficients(
    data: np.ndarray, n_sub: int = None, bandwidth: float = 0.5,
) -> dict:
    if n_sub is None:
        n_sub = N_KERNEL
    N, p = data.shape
    n_sub = min(n_sub, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)
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
    coeff_map = {}
    for j in range(p):
        c_all, other_cols = all_coeffs[j]
        for idx_in_other, k in enumerate(other_cols):
            coeff_sub = c_all[:, idx_in_other + 1]
            coeff_map[(k, j)] = coeff_sub[nearest].astype(np.float32)
    return coeff_map


def build_edge_tensor(df: pd.DataFrame) -> tuple:
    """
    Build multi-bandwidth edge tensor.

    Channels per edge (u→v):
      Ch 0: sorted u observations
      Ch 1: v observations sorted by u
      Ch 2: kernel regression coeff at bandwidth=0.2 (sharp/local)
      Ch 3: kernel regression coeff at bandwidth=0.5 (medium)
      Ch 4: kernel regression coeff at bandwidth=1.0 (smooth/global)

    Returns:
      edge_data:   (E, N_CHANNELS, N) float32
      edge_types:  (E,) int64
    """
    cols = list(df.columns)
    p = len(cols)
    data = df.values.astype(np.float32)
    N = data.shape[0]

    # Compute kernel regression at each bandwidth
    # Share the same subsample across bandwidths for consistency
    n_sub = min(N_KERNEL, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)

    coeff_maps = []
    for bw in BANDWIDTHS:
        # Use same subsample for all bandwidths
        data_sub = data[sub_idx]
        diff = data_sub[:, None, :] - data_sub[None, :, :]
        sq_dist = (diff ** 2).sum(axis=-1)
        W = np.exp(-sq_dist / (2 * bw ** 2))

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

        coeff_map = {}
        for j in range(p):
            c_all, other_cols = all_coeffs[j]
            for idx_in_other, k in enumerate(other_cols):
                coeff_sub = c_all[:, idx_in_other + 1]
                coeff_map[(k, j)] = coeff_sub[nearest].astype(np.float32)
        coeff_maps.append(coeff_map)

    edges = []
    edge_types = []

    for i, u_name in enumerate(cols):
        sort_idx = np.argsort(data[:, i])
        u_sorted = data[sort_idx, i]

        for j, v_name in enumerate(cols):
            if i == j:
                continue

            v_sorted_by_u = data[sort_idx, j]

            # Stack: [sorted_u, sorted_v, coeff_bw0, coeff_bw1, coeff_bw2]
            channels = [u_sorted, v_sorted_by_u]
            for cm in coeff_maps:
                channels.append(cm[(i, j)][sort_idx])

            edge_tensor = np.stack(channels, axis=0)  # (N_CHANNELS, N)
            edges.append(edge_tensor)
            edge_types.append(_edge_type(u_name, v_name))

    edge_data = np.stack(edges, axis=0).astype(np.float32)  # (E, N_CHANNELS, N)
    edge_types = np.array(edge_types, dtype=np.int64)
    return edge_data, edge_types


def _build_single(args):
    df, y_df = args
    edge_data, edge_types = build_edge_tensor(df)
    cols = list(df.columns)
    p = len(cols)
    result = {
        "edge_data": edge_data, "edge_types": edge_types,
        "cols": cols, "p": p,
    }
    if y_df is not None:
        adj_np = y_df.values.astype(np.float32)
        adj_cols = list(y_df.columns)
        result["adj"] = adj_np
        edge_labels = []
        for ui in range(p):
            for vi in range(p):
                if ui != vi:
                    edge_labels.append(int(adj_np[ui, vi]))
        result["edge_labels"] = np.array(edge_labels, dtype=np.int64)
        _, adjacency_label = create_graph_label()
        df_adj = pd.DataFrame(adj_np, columns=adj_cols, index=adj_cols)
        other_nodes = [v for v in adj_cols if v not in ("X", "Y")]
        node_labels = []
        for v in other_nodes:
            sub = df_adj.loc[[v, "X", "Y"], [v, "X", "Y"]]
            key = tuple(sub.values.flatten())
            label_str = adjacency_label.get(key, "Independent")
            node_labels.append(CLASS_NAMES.index(label_str))
        result["node_labels"] = np.array(node_labels, dtype=np.int64)
        result["other_nodes"] = other_nodes
    return result


# ============================================================
# Dataset & Collation
# ============================================================
class CausalEdgeDataset(Dataset):
    def __init__(self, X_list, y_list=None, n_workers=None, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}...")
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
            print(f"Loaded {len(self.samples)} samples from cache.")
            return
        if n_workers is None:
            import multiprocessing as mp
            n_workers = max(1, mp.cpu_count() - 1)
        args = [(X_list[i], y_list[i] if y_list else None) for i in range(len(X_list))]
        print(f"Building edge dataset ({len(args)} samples, {n_workers} workers)...")
        if n_workers > 1:
            raw = [None] * len(args)
            ctx = __import__('multiprocessing').get_context('fork')
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {pool.submit(_build_single, a): idx for idx, a in enumerate(args)}
                for fut in tqdm(as_completed(futures), total=len(args)):
                    raw[futures[fut]] = fut.result()
        else:
            raw = [_build_single(a) for a in tqdm(args)]

        self.samples = raw  # store as numpy

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            print(f"Saving dataset cache to {cache_path}...")
            with open(cache_path, "wb") as f:
                pickle.dump(self.samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cached {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        sample = {
            "edge_data":  torch.from_numpy(item["edge_data"]),
            "edge_types": torch.from_numpy(item["edge_types"]),
            "cols": item["cols"], "p": item["p"],
        }
        if "adj" in item:
            sample["adj"]         = torch.from_numpy(item["adj"])
            sample["edge_labels"] = torch.from_numpy(item["edge_labels"])
            sample["node_labels"] = torch.from_numpy(item["node_labels"])
            sample["other_nodes"] = item["other_nodes"]
        return sample


def collate_fn(batch):
    max_E = max(item["edge_data"].shape[0] for item in batch)
    B = len(batch)

    edge_data  = torch.zeros(B, max_E, N_CHANNELS, N_OBS)
    edge_types = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask  = torch.zeros(B, max_E, dtype=torch.bool)

    max_K = max((item["node_labels"].shape[0] if "node_labels" in item else 0) for item in batch)
    edge_labels = torch.full((B, max_E), -1, dtype=torch.long)
    node_labels = torch.full((B, max_K), -1, dtype=torch.long)
    node_mask   = torch.zeros(B, max_K, dtype=torch.bool)

    cols_list = []
    has_labels = False

    for b, item in enumerate(batch):
        E = item["edge_data"].shape[0]
        edge_data[b, :E]  = item["edge_data"]
        edge_types[b, :E] = item["edge_types"]
        edge_mask[b, :E]  = True
        cols_list.append(item["cols"])
        if "edge_labels" in item:
            has_labels = True
            edge_labels[b, :E] = item["edge_labels"]
            K = item["node_labels"].shape[0]
            node_labels[b, :K] = item["node_labels"]
            node_mask[b, :K] = True

    out = {"edge_data": edge_data, "edge_types": edge_types,
           "edge_mask": edge_mask, "cols": cols_list}
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels
        out["node_mask"]   = node_mask
    return out


# ============================================================
# Model (v2 architecture, just StemLayer takes N_CHANNELS)
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, d, kernel_size=3, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(n_groups, d)
        self.act = nn.GELU()
    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))

class MergeOperator(nn.Module):
    def __init__(self, n_inputs, d):
        super().__init__()
        self.linear = nn.Linear(n_inputs * d, d)
        self.norm = nn.LayerNorm(d)
        self.act = nn.GELU()
    def forward(self, inputs):
        return self.act(self.norm(self.linear(torch.cat(inputs, dim=-1))))

class StemLayer(nn.Module):
    def __init__(self, d, n_channels=None):
        super().__init__()
        n_channels = n_channels or N_CHANNELS
        self.linear = nn.Linear(n_channels, d)
    def forward(self, x):
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

class EdgeFeatureExtractor(nn.Module):
    def __init__(self, d=64, n_blocks=5):
        super().__init__()
        self.stem = StemLayer(d)
        self.blocks = nn.Sequential(*[ConvBlock(d) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)

class SelfAttentionLayer(nn.Module):
    def __init__(self, d=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm2 = nn.LayerNorm(d)
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class ADIAModel(nn.Module):
    """
    v2 architecture with multi-bandwidth channels.

    Only change: StemLayer accepts N_CHANNELS (5) instead of 3.
    The rest is identical to v2. 2-input merge (conv + type).
    """
    def __init__(self, d=None, n_edge_types=None):
        super().__init__()
        d = d or D_MODEL
        n_edge_types = n_edge_types or N_EDGE_TYPES
        self.d = d

        self.extractor = EdgeFeatureExtractor(d)
        self.edge_type_emb = nn.Embedding(n_edge_types, d)
        self.edge_merge = MergeOperator(n_inputs=2, d=d)
        self.attn_layers = nn.ModuleList([SelfAttentionLayer(d) for _ in range(2)])
        self.edge_head = nn.Linear(d, 2)
        self.node_merge = MergeOperator(n_inputs=4, d=d)
        self.node_head = nn.Linear(d, N_CLASSES)

    def forward(self, edge_data, edge_types, edge_mask, cols_list):
        B, E, C, N = edge_data.shape

        edge_emb = self.extractor(edge_data.view(B * E, C, N)).view(B, E, self.d)
        type_emb = self.edge_type_emb(edge_types)
        edge_emb = self.edge_merge([edge_emb, type_emb])

        pad_mask = ~edge_mask
        for attn in self.attn_layers:
            edge_emb = attn(edge_emb, key_padding_mask=pad_mask)

        edge_logits = self.edge_head(edge_emb)

        node_logits_list = []
        for b in range(B):
            cols = cols_list[b]
            p = len(cols)
            col_idx = {name: i for i, name in enumerate(cols)}
            edge_order = {}
            count = 0
            for ui in range(p):
                for vi in range(p):
                    if ui != vi:
                        edge_order[(ui, vi)] = count
                        count += 1
            x_idx, y_idx = col_idx.get("X"), col_idx.get("Y")
            embs = edge_emb[b]
            other_nodes = [n for n in cols if n not in ("X", "Y")]
            if not other_nodes or x_idx is None or y_idx is None:
                node_logits_list.append(None)
                continue
            node_logits = []
            for node_name in other_nodes:
                u = col_idx[node_name]
                gathered = [
                    embs[edge_order[(u, x_idx)]],
                    embs[edge_order[(u, y_idx)]],
                    embs[edge_order[(x_idx, u)]],
                    embs[edge_order[(y_idx, u)]],
                ]
                node_emb = self.node_merge(gathered)
                node_logits.append(self.node_head(node_emb))
            node_logits_list.append(
                torch.stack(node_logits) if node_logits else None
            )
        return edge_logits, node_logits_list


# ============================================================
# Training Wrapper (v2 style, no stat injection)
# ============================================================
def compute_class_weights(y_list):
    adjacency_label = get_adjacency_label()
    counts = torch.zeros(N_CLASSES)
    for y_df in y_list:
        cols = list(y_df.columns)
        arr = y_df.values
        col_idx = {c: i for i, c in enumerate(cols)}
        for v in cols:
            if v in ("X", "Y"): continue
            idx = [col_idx[v], col_idx["X"], col_idx["Y"]]
            sub = arr[np.ix_(idx, idx)]
            key = tuple(sub.flatten())
            label_str = adjacency_label.get(key, "Independent")
            counts[CLASS_NAMES.index(label_str)] += 1
    w = 1.0 / (counts + 1e-6)
    return w / w.sum() * N_CLASSES

def compute_edge_class_weights(y_list):
    counts = torch.zeros(2)
    for y_df in y_list:
        arr = y_df.values
        p = arr.shape[0]
        for i in range(p):
            for j in range(p):
                if i != j:
                    counts[int(arr[i, j])] += 1
    w = 1.0 / (counts + 1e-6)
    return w / w.sum() * 2


class ADIAModelWrapper(pl.LightningModule):
    def __init__(self, d=None, node_class_weights=None,
                 edge_class_weights=None, lr=1e-3, max_epochs=20):
        super().__init__()
        d = d or D_MODEL
        self.model = ADIAModel(d)
        self.lr = lr
        self.max_epochs = max_epochs
        self.node_criterion = nn.CrossEntropyLoss(
            weight=node_class_weights if node_class_weights is not None
            else torch.ones(N_CLASSES), ignore_index=-1)
        self.edge_criterion = nn.CrossEntropyLoss(
            weight=edge_class_weights if edge_class_weights is not None
            else torch.ones(2), ignore_index=-1)

    def forward(self, batch):
        return self.model(
            batch["edge_data"].to(self.device),
            batch["edge_types"].to(self.device),
            batch["edge_mask"].to(self.device),
            batch["cols"],
        )

    def _compute_loss(self, batch, split):
        edge_logits, node_logits_list = self(batch)
        B = edge_logits.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)
        n_terms = 0
        if "edge_labels" in batch:
            el = batch["edge_labels"].to(self.device)
            edge_loss = self.edge_criterion(edge_logits.view(-1, 2), el.view(-1))
            total_loss = total_loss + edge_loss
            n_terms += 1
        if "node_labels" in batch:
            nl = batch["node_labels"].to(self.device)
            all_logits, all_labels = [], []
            for b in range(B):
                if node_logits_list[b] is not None:
                    K = node_logits_list[b].shape[0]
                    all_logits.append(node_logits_list[b])
                    all_labels.append(nl[b, :K])
            if all_logits:
                cat_logits = torch.cat(all_logits, dim=0)
                cat_labels = torch.cat(all_labels, dim=0)
                node_loss = self.node_criterion(cat_logits, cat_labels)
                total_loss = total_loss + node_loss
                n_terms += 1
        if n_terms > 0:
            total_loss = total_loss / n_terms
        self.log(f"{split}_loss", total_loss, on_step=(split == "train"),
                 on_epoch=True, prog_bar=True, batch_size=B)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def infer_single(df, model, device="cpu"):
    model = model.eval()
    cols = list(df.columns)
    p = len(cols)
    edge_data, edge_types = build_edge_tensor(df)
    edge_data_t  = torch.tensor(edge_data).unsqueeze(0).to(device)
    edge_types_t = torch.tensor(edge_types).unsqueeze(0).to(device)
    edge_mask_t  = torch.ones(1, len(edge_types), dtype=torch.bool, device=device)
    edge_logits, node_logits_list = model(
        edge_data_t, edge_types_t, edge_mask_t, [cols]
    )
    edge_probs = torch.softmax(edge_logits[0], dim=-1)[:, 1]
    E_mat = np.zeros((p, p))
    count = 0
    for i in range(p):
        for j in range(p):
            if i != j:
                E_mat[i, j] = edge_probs[count].item()
                count += 1
    adj = transform_proba_to_DAG(cols, E_mat).astype(int)
    A = pd.DataFrame(adj, columns=cols, index=cols)
    if node_logits_list[0] is not None:
        other_nodes = [n for n in cols if n not in ("X", "Y")]
        node_preds = torch.argmax(node_logits_list[0], dim=-1)
        patterns = {
            "Confounder":        lambda n: [(n, "X"), (n, "Y")],
            "Collider":          lambda n: [("X", n), ("Y", n)],
            "Mediator":          lambda n: [("X", n), (n, "Y")],
            "Cause of X":        lambda n: [(n, "X")],
            "Cause of Y":        lambda n: [(n, "Y")],
            "Consequence of X":  lambda n: [("X", n)],
            "Consequence of Y":  lambda n: [("Y", n)],
            "Independent":       lambda n: [],
        }
        for k, node_name in enumerate(other_nodes):
            pred_class = CLASS_NAMES[node_preds[k].item()]
            A.loc[node_name, :] = 0
            A.loc[:, node_name] = 0
            for (src, dst) in patterns[pred_class](node_name):
                A.loc[src, dst] = 1
    return A


@torch.no_grad()
def infer_batch_local(dfs, model, device="cpu", batch_size=32, cache_dir=None):
    model = model.eval()
    cache_path = None
    if cache_dir:
        bw_tag = "_".join(f"{b}" for b in BANDWIDTHS)
        cache_path = os.path.join(cache_dir, f"infer_edge_tensors_multibw_{bw_tag}_nk{N_KERNEL}.pkl")
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached edge tensors from {cache_path}...")
        with open(cache_path, "rb") as f:
            all_items = pickle.load(f)
        print(f"Loaded {len(all_items)} cached edge tensors.")
    else:
        import multiprocessing as mp
        n_workers = max(1, mp.cpu_count() - 1)
        args = [(df, None) for df in dfs]
        print(f"Building edge tensors ({len(dfs)} graphs, {n_workers} workers)...")
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            raw = list(tqdm(pool.map(_build_single, args, chunksize=8), total=len(args)))
        all_items = [{
            "edge_data": torch.from_numpy(item["edge_data"]),
            "edge_types": torch.from_numpy(item["edge_types"]),
            "cols": item["cols"], "p": item["p"],
        } for item in raw]
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(all_items, f, protocol=pickle.HIGHEST_PROTOCOL)
    results = [None] * len(all_items)
    patterns = {
        "Confounder":        lambda n: [(n, "X"), (n, "Y")],
        "Collider":          lambda n: [("X", n), ("Y", n)],
        "Mediator":          lambda n: [("X", n), (n, "Y")],
        "Cause of X":        lambda n: [(n, "X")],
        "Cause of Y":        lambda n: [(n, "Y")],
        "Consequence of X":  lambda n: [("X", n)],
        "Consequence of Y":  lambda n: [("Y", n)],
        "Independent":       lambda n: [],
    }
    for start in tqdm(range(0, len(all_items), batch_size), desc="infering batch"):
        batch_items = all_items[start:start + batch_size]
        batch = collate_fn(batch_items)
        edge_logits, node_logits_list = model(
            batch["edge_data"].to(device), batch["edge_types"].to(device),
            batch["edge_mask"].to(device), batch["cols"],
        )
        for b, item in enumerate(batch_items):
            cols, p, idx = item["cols"], item["p"], start + b
            edge_probs = torch.softmax(edge_logits[b], dim=-1)[:, 1]
            E_mat = np.zeros((p, p))
            count = 0
            for ii in range(p):
                for jj in range(p):
                    if ii != jj:
                        E_mat[ii, jj] = edge_probs[count].item()
                        count += 1
            adj = transform_proba_to_DAG(cols, E_mat).astype(int)
            A = pd.DataFrame(adj, columns=cols, index=cols)
            if node_logits_list[b] is not None:
                other_nodes = [n for n in cols if n not in ("X", "Y")]
                node_preds = torch.argmax(node_logits_list[b], dim=-1)
                for k, node_name in enumerate(other_nodes):
                    pred_class = CLASS_NAMES[node_preds[k].item()]
                    A.loc[node_name, :] = 0
                    A.loc[:, node_name] = 0
                    for (src, dst) in patterns[pred_class](node_name):
                        A.loc[src, dst] = 1
            results[idx] = A
    return results


# ============================================================
# Train & Infer
# ============================================================
def train(
    X_train: typing.Dict[str, pd.DataFrame],
    y_train: typing.Dict[str, pd.DataFrame],
    model_directory_path: str,
) -> None:
    keys = list(X_train.keys())
    X_list = [X_train[k] for k in keys]
    y_list = [y_train[k] for k in keys]

    node_w_path = os.path.join(LOCAL_CACHE_DIR, "node_weights.pt")
    edge_w_path = os.path.join(LOCAL_CACHE_DIR, "edge_weights.pt")
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    if os.path.exists(node_w_path) and os.path.exists(edge_w_path):
        print("Loading cached class weights...")
        node_w = torch.load(node_w_path, weights_only=True)
        edge_w = torch.load(edge_w_path, weights_only=True)
    else:
        print("Computing class weights...")
        node_w = compute_class_weights(y_list)
        edge_w = compute_edge_class_weights(y_list)
        torch.save(node_w, node_w_path)
        torch.save(edge_w, edge_w_path)

    bw_tag = "_".join(f"{b}" for b in BANDWIDTHS)
    dataset_path = os.path.join(
        LOCAL_CACHE_DIR, f"train_dataset_multibw_{bw_tag}_nk{N_KERNEL}.pkl"
    )
    dataset = CausalEdgeDataset(X_list, y_list, cache_path=dataset_path, n_workers=47)

    wrapper = ADIAModelWrapper(
        d=D_MODEL, node_class_weights=node_w,
        edge_class_weights=edge_w, lr=1e-3, max_epochs=MAX_EPOCHS,
    )

    total_params = sum(p.numel() for p in wrapper.model.parameters())
    print(f"Model params: {total_params:,}")
    print(f"Config: bandwidths={BANDWIDTHS}, channels={N_CHANNELS}")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=2, max_epochs=MAX_EPOCHS, precision="32-true",
        logger=True, enable_checkpointing=True, enable_progress_bar=True,
    )
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, drop_last=False,
        num_workers=5, collate_fn=collate_fn,
    )
    print("Starting training...")
    trainer.fit(wrapper, loader)

    path = os.path.join(model_directory_path, "model.pt")
    torch.save(wrapper.model.state_dict(), path)
    print(f"Model saved to {path}")


def infer(
    X_test: typing.Dict[str, pd.DataFrame],
    model_directory_path: str,
    id_column_name: str,
    prediction_column_name: str,
) -> pd.DataFrame:
    path = os.path.join(model_directory_path, "model_v5_multibw.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()

    names = list(X_test.keys())
    dfs = [X_test[n] for n in names]

    print(f"Batch inference on {len(dfs)} samples (device={device})...")
    adj_list = infer_batch_local(dfs, model, device=device, batch_size=64,
                                 cache_dir=LOCAL_CACHE_DIR)

    submission = {}
    for name, A in zip(names, adj_list):
        nodes = list(A.columns)
        for i in nodes:
            for j in nodes:
                submission[f"{name}_{i}_{j}"] = int(A.loc[i, j])

    s = pd.Series(submission).reset_index()
    s.columns = [id_column_name, prediction_column_name]
    return s


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # X_train = pd.read_pickle("data/X_train.pickle")
    # y_train = pd.read_pickle("data/y_train.pickle")
    # print(f"Loaded {len(X_train)} training samples.")

    # train(X_train, y_train, model_directory_path="resources")

    # Local evaluation
    X_test = pd.read_pickle("data/X_test_reduced.pickle")
    y_test = pd.read_pickle("data/y_test_reduced.pickle")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL)
    model.load_state_dict(torch.load("resources/model_v5_multibw.pt", map_location=device, weights_only=True))
    model.to(device).eval()

    names = list(X_test.keys())
    dfs = [X_test[n] for n in names]
    adj_list = infer_batch_local(dfs, model, device=device, batch_size=64,
      cache_dir=LOCAL_CACHE_DIR)

    adjacency_label = get_adjacency_label()
    class_correct = {c: 0 for c in CLASS_NAMES}
    class_total = {c: 0 for c in CLASS_NAMES}

    for name, A_pred in zip(names, adj_list):
        y_df = y_test[name]
        pred_labels = get_labels(A_pred, adjacency_label)
        true_labels = get_labels(y_df, adjacency_label)
        for v in true_labels:
            true_cls = true_labels[v]
            pred_cls = pred_labels.get(v, "Independent")
            class_total[true_cls] += 1
            if pred_cls == true_cls:
                class_correct[true_cls] += 1

    print("\nPer-class accuracy:")
    accs = []
    for cls in CLASS_NAMES:
        n = class_total[cls]
        acc = class_correct[cls] / n if n > 0 else 0.0
        accs.append(acc)
        print(f"  {cls:25s}: {acc:.4f}  (n={n})")
    print(f"\nBalanced Accuracy: {np.mean(accs):.4f}")
