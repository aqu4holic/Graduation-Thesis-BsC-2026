# %% [markdown]
# # Adia Lab Causal Discovery — 1st Place Solution
#
# Faithful reimplementation of [thetourney.github.io/adia-report](https://thetourney.github.io/adia-report/).
#
# **Key difference from a naive approach**: the 3rd channel is a **multivariate**
# kernel regression coefficient — predicting variable j from *all* other variables
# simultaneously (not just pairwise), using a Gaussian kernel on full-row distance.
# This captures conditional dependencies critical for causal discovery.

# %% [markdown]
# ### Setup

# %% [markdown]
# ### Imports

# %%
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

# %% [markdown]
# ### Configuration

# %%
# @crunch/keep:on
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

# === v5 config ===
N_EDGE_STATS = 2   # partial_corr, residual_asym (per edge scalars)
AUG_NOISE_STD = 0.01  # Gaussian noise added to edge features during training


# %% [markdown]
# ### Graph Utilities

# %%
# @crunch/keep:on

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

# %% [markdown]
# ### Data Preprocessing
#
# **v5 changes from v2 baseline:**
# 1. `build_edge_tensor` now also computes **edge-level statistics** (partial correlation,
#    residual asymmetry) as scalar features per edge. These are NOT used as conv channels —
#    they're injected after conv pooling in the merge step.
# 2. Light **Gaussian noise augmentation** on edge features during training.
#
# The 3-channel conv input is unchanged from v2 (sorted_u, sorted_v, kernel_coeff).
#

# %%
def _edge_type(u_name, v_name):
    """Edge type encoding (7 types) as described in the report."""
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
) -> np.ndarray:
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


def compute_edge_statistics(data: np.ndarray) -> tuple:
    """
    Compute per-edge scalar statistics for stat injection.

    Returns:
        pcorr_matrix:  (p, p) partial correlation via precision matrix
        resid_matrix:  (p, p) residual-cause correlation for direction asymmetry
    """
    N, p = data.shape

    # --- Partial correlation via precision matrix ---
    cov = np.cov(data.T)
    cov += 1e-6 * np.eye(p)
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        precision = np.eye(p)

    diag = np.sqrt(np.maximum(np.diag(precision), 1e-10))
    pcorr = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcorr, 0.0)

    # --- Residual asymmetry via linear regression ---
    resid_corr = np.zeros((p, p), dtype=np.float32)
    for i in range(p):
        xi = data[:, i]
        xi_var = np.var(xi)
        if xi_var < 1e-10:
            continue
        xi_mean = xi.mean()
        for j in range(p):
            if i == j:
                continue
            xj = data[:, j]
            b = np.cov(xi, xj)[0, 1] / (xi_var + 1e-10)
            a = xj.mean() - b * xi_mean
            residuals = xj - (a + b * xi)
            r_std = np.std(residuals)
            if r_std < 1e-10:
                resid_corr[i, j] = 0.0
            else:
                resid_corr[i, j] = np.corrcoef(residuals, xi)[0, 1]

    return pcorr.astype(np.float32), resid_corr


def build_edge_tensor(df: pd.DataFrame) -> tuple:
    """
    Build edge data tensor, edge types, AND edge-level statistics.

    Returns:
      edge_data:   (E, 3, N) float32      — 3-channel sorted features (unchanged)
      edge_types:  (E,) int64
      edge_stats:  (E, N_EDGE_STATS) float32 — per-edge scalar statistics
    """
    cols = list(df.columns)
    p = len(cols)
    data = df.values.astype(np.float32)
    N = data.shape[0]

    coeff_map = compute_multivariate_kernel_coefficients(data)
    pcorr_matrix, resid_matrix = compute_edge_statistics(data)

    edges, edge_types, edge_stats = [], [], []

    for i, u_name in enumerate(cols):
        sort_idx = np.argsort(data[:, i])
        u_sorted = data[sort_idx, i]

        for j, v_name in enumerate(cols):
            if i == j:
                continue

            v_sorted_by_u = data[sort_idx, j]
            coeff_sorted = coeff_map[(i, j)][sort_idx]

            edge_tensor = np.stack([u_sorted, v_sorted_by_u, coeff_sorted], axis=0)
            edges.append(edge_tensor)
            edge_types.append(_edge_type(u_name, v_name))

            # Per-edge scalar stats
            stats = np.array([
                pcorr_matrix[i, j],
                resid_matrix[i, j],
            ], dtype=np.float32)
            edge_stats.append(stats)

    edge_data  = np.stack(edges, axis=0).astype(np.float32)
    edge_types = np.array(edge_types, dtype=np.int64)
    edge_stats = np.stack(edge_stats, axis=0).astype(np.float32)
    return edge_data, edge_types, edge_stats


def _build_single(args):
    df, y_df = args
    edge_data, edge_types, edge_stats = build_edge_tensor(df)
    cols = list(df.columns)
    p = len(cols)
    result = {
        "edge_data": edge_data, "edge_types": edge_types,
        "edge_stats": edge_stats, "cols": cols, "p": p,
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


# %% [markdown]
# ### Dataset & Collation

# %%
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        self.samples = []
        for item in raw:
            sample = {
                "edge_data":  torch.from_numpy(item["edge_data"]),
                "edge_types": torch.from_numpy(item["edge_types"]),
                "edge_stats": torch.from_numpy(item["edge_stats"]),
                "cols": item["cols"], "p": item["p"],
            }
            if "adj" in item:
                sample["adj"]         = torch.from_numpy(item["adj"])
                sample["edge_labels"] = torch.from_numpy(item["edge_labels"])
                sample["node_labels"] = torch.from_numpy(item["node_labels"])
                sample["other_nodes"] = item["other_nodes"]
            self.samples.append(sample)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            print(f"Saving dataset cache to {cache_path}...")
            with open(cache_path, "wb") as f:
                pickle.dump(self.samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cached {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    max_E = max(item["edge_data"].shape[0] for item in batch)
    B = len(batch)
    n_stats = batch[0]["edge_stats"].shape[1]

    edge_data  = torch.zeros(B, max_E, 3, N_OBS)
    edge_types = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask  = torch.zeros(B, max_E, dtype=torch.bool)
    edge_stats = torch.zeros(B, max_E, n_stats)

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
        edge_stats[b, :E] = item["edge_stats"]
        cols_list.append(item["cols"])
        if "edge_labels" in item:
            has_labels = True
            edge_labels[b, :E] = item["edge_labels"]
            K = item["node_labels"].shape[0]
            node_labels[b, :K] = item["node_labels"]
            node_mask[b, :K] = True

    out = {"edge_data": edge_data, "edge_types": edge_types,
           "edge_mask": edge_mask, "edge_stats": edge_stats, "cols": cols_list}
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels
        out["node_mask"]   = node_mask
    return out


# %% [markdown]
# ### Model Architecture — v5
#
# **v5 = v2 baseline + edge-level stat injection + training noise augmentation.**
#
# Changes from v2:
# 1. **Edge statistics** (partial corr, residual asymmetry) are computed as scalars per edge
#    during preprocessing. A small MLP projects these into d-dim, then they're merged
#    alongside the edge type embedding: `MergeOperator([conv_emb, type_emb, stat_emb])`.
# 2. **Gaussian noise** (std=0.01) is added to edge features during training for regularization.
#
# The 3-channel conv path is completely unchanged from v2.
# The merge step goes from 2-input to 3-input (conv + type + stats).
#

# %%
class ConvBlock(nn.Module):
    """Residual Conv1d + GroupNorm + GELU (Fig. 5 in report)."""
    def __init__(self, d, kernel_size=3, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(n_groups, d)
        self.act = nn.GELU()
    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))

class MergeOperator(nn.Module):
    """Concat inputs → Linear → LayerNorm → GELU (Fig. 6 in report)."""
    def __init__(self, n_inputs, d):
        super().__init__()
        self.linear = nn.Linear(n_inputs * d, d)
        self.norm = nn.LayerNorm(d)
        self.act = nn.GELU()
    def forward(self, inputs):
        return self.act(self.norm(self.linear(torch.cat(inputs, dim=-1))))

class StemLayer(nn.Module):
    """Linear 3 → d_model applied per-observation."""
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(3, d)
    def forward(self, x):
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

class EdgeFeatureExtractor(nn.Module):
    """Stem + 5×ConvBlock + AvgPool → 64-dim embedding."""
    def __init__(self, d=64, n_blocks=5):
        super().__init__()
        self.stem = StemLayer(d)
        self.blocks = nn.Sequential(*[ConvBlock(d) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


class StatProjector(nn.Module):
    """
    Project per-edge scalar statistics into d-dim embedding.

    Takes (partial_corr, residual_asym) per edge → MLP → d-dim.
    Injected at the merge step alongside conv_emb and type_emb.
    """
    def __init__(self, n_stats, d):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_stats, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, stats):
        return self.proj(stats)


class SelfAttentionLayer(nn.Module):
    """Standard multi-head self-attention + FFN with pre-norm residuals."""
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
    v5: Baseline + edge-level stat injection + training noise.

    Pipeline:
      1. EdgeFeatureExtractor: (B,E,3,N) → (B,E,d)        [unchanged]
      2. StatProjector: (B,E,2) → (B,E,d)                  [NEW]
      3. EdgeTypeMerge: [conv_emb, type_emb, stat_emb] → d  [3-input merge]
      4. 2× SelfAttention
      5. Edge head + Node head

    Training augmentation: Gaussian noise on edge features.
    """
    def __init__(self, d=None, n_edge_types=None, n_edge_stats=None,
                 aug_noise_std=0.0):
        super().__init__()
        d = d or D_MODEL
        n_edge_types = n_edge_types or N_EDGE_TYPES
        n_edge_stats = n_edge_stats or N_EDGE_STATS

        self.d = d
        self.aug_noise_std = aug_noise_std

        self.extractor = EdgeFeatureExtractor(d)
        self.stat_proj = StatProjector(n_edge_stats, d)
        self.edge_type_emb = nn.Embedding(n_edge_types, d)

        # 3-input merge: conv + type + stats
        self.edge_merge = MergeOperator(n_inputs=3, d=d)

        self.attn_layers = nn.ModuleList([SelfAttentionLayer(d) for _ in range(2)])
        self.edge_head = nn.Linear(d, 2)
        self.node_merge = MergeOperator(n_inputs=4, d=d)
        self.node_head = nn.Linear(d, N_CLASSES)

    def forward(self, edge_data, edge_types, edge_mask, cols_list, edge_stats=None):
        B, E, C, N = edge_data.shape

        # --- Training augmentation: add Gaussian noise to edge features ---
        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std

        # --- Conv path (unchanged from v2) ---
        edge_emb = self.extractor(edge_data.view(B * E, C, N)).view(B, E, self.d)

        # --- Edge type embedding ---
        type_emb = self.edge_type_emb(edge_types)

        # --- Stat projection (NEW) ---
        if edge_stats is not None:
            stat_emb = self.stat_proj(edge_stats)
        else:
            stat_emb = torch.zeros_like(edge_emb)

        # --- 3-way merge ---
        edge_emb = self.edge_merge([edge_emb, type_emb, stat_emb])

        # --- Self-attention ---
        pad_mask = ~edge_mask
        for attn in self.attn_layers:
            edge_emb = attn(edge_emb, key_padding_mask=pad_mask)

        # --- Edge head ---
        edge_logits = self.edge_head(edge_emb)

        # --- Node head ---
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


# %% [markdown]
# ### Training Wrapper
#
# Loss = CE(edge classification) + CE(node classification),
# both weighted by inverse class frequency as described in the report.

# %%
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
                 edge_class_weights=None, lr=1e-3, max_epochs=20,
                 aug_noise_std=0.0):
        super().__init__()
        d = d or D_MODEL
        self.model = ADIAModel(d, aug_noise_std=aug_noise_std)
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
            edge_stats=batch["edge_stats"].to(self.device),
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


# %% [markdown]
# ### Inference

# %%
@torch.no_grad()
def infer_single(df, model, device="cpu"):
    model = model.eval()
    cols = list(df.columns)
    p = len(cols)
    edge_data, edge_types, edge_stats = build_edge_tensor(df)
    edge_data_t  = torch.tensor(edge_data).unsqueeze(0).to(device)
    edge_types_t = torch.tensor(edge_types).unsqueeze(0).to(device)
    edge_mask_t  = torch.ones(1, len(edge_types), dtype=torch.bool, device=device)
    edge_stats_t = torch.tensor(edge_stats).unsqueeze(0).to(device)
    edge_logits, node_logits_list = model(
        edge_data_t, edge_types_t, edge_mask_t, [cols], edge_stats=edge_stats_t
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
def infer_batch(dfs, model, device="cpu", batch_size=32):
    model = model.eval()
    from concurrent.futures import ProcessPoolExecutor
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
        "edge_stats": torch.from_numpy(item["edge_stats"]),
        "cols": item["cols"], "p": item["p"],
    } for item in raw]
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
            edge_stats=batch["edge_stats"].to(device),
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

@torch.no_grad()
def infer_batch_local(dfs, model, device="cpu", batch_size=32, cache_dir=None):
    model = model.eval()
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"infer_edge_tensors_v5_nk{N_KERNEL}.pkl")
    if cache_path and os.path.exists(cache_path):
        import pickle
        print(f"Loading cached edge tensors from {cache_path}...")
        with open(cache_path, "rb") as f:
            all_items = pickle.load(f)
        print(f"Loaded {len(all_items)} cached edge tensors.")
    else:
        from concurrent.futures import ProcessPoolExecutor
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
            "edge_stats": torch.from_numpy(item["edge_stats"]),
            "cols": item["cols"], "p": item["p"],
        } for item in raw]
        if cache_path:
            import pickle
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
            edge_stats=batch["edge_stats"].to(device),
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


# %% [markdown]
# ### CrunchDAO Interface — `train` & `infer`

# %%
# @crunch/keep:on
MAX_EPOCHS = 30
LOCAL_CACHE_DIR = "dataset_cache/"

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

    dataset_path = os.path.join(LOCAL_CACHE_DIR, f"train_dataset_v5_nk{N_KERNEL}.pkl")
    dataset = CausalEdgeDataset(X_list, y_list, cache_path=dataset_path, n_workers=40)

    wrapper = ADIAModelWrapper(
        d=D_MODEL, node_class_weights=node_w,
        edge_class_weights=edge_w, lr=1e-3, max_epochs=MAX_EPOCHS,
        aug_noise_std=AUG_NOISE_STD,
    )

    total_params = sum(p.numel() for p in wrapper.model.parameters())
    trainable_params = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Config: stat_injection=ON, aug_noise={AUG_NOISE_STD}")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=2, max_epochs=MAX_EPOCHS, precision="32-true",
        logger=True, enable_checkpointing=True, enable_progress_bar=True,
    )
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, drop_last=False,
        num_workers=10, collate_fn=collate_fn,
    )
    print("Starting training...")
    trainer.fit(wrapper, loader)

    path = os.path.join(model_directory_path, "model_v5_augstat.pt")
    torch.save(wrapper.model.state_dict(), path)
    print(f"Model saved to {path}")

def infer(
    X_test: typing.Dict[str, pd.DataFrame],
    model_directory_path: str,
    id_column_name: str,
    prediction_column_name: str,
) -> pd.DataFrame:
    path = os.path.join(model_directory_path, "model_v5_augstat.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL, aug_noise_std=0.0)  # no noise at inference
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()

    y_test = pd.read_pickle("data/y_test_reduced.pickle")
    names = list(X_test.keys())[:10]
    dfs = [X_test[n] for n in names]
    adj_list = infer_batch_local(dfs, model, device=device, batch_size=10, cache_dir=None)  # no cache!

    adjacency_label = get_adjacency_label()
    for name, A_pred in zip(names, adj_list):
        y_df = y_test[name]
        pred_labels = get_labels(A_pred, adjacency_label)
        true_labels = get_labels(y_df, adjacency_label)
        for v in true_labels:
            print(f"  {name}/{v}: true={true_labels[v]:20s} pred={pred_labels.get(v, '???')}")

    # print(f"Batch inference on {len(dfs)} samples (device={device})...")
    # adj_list = infer_batch_local(dfs, model, device=device, batch_size=64,
    #                        cache_dir=LOCAL_CACHE_DIR)

    # submission = {}
    # for name, A in zip(names, adj_list):
    #     nodes = list(A.columns)
    #     for i in nodes:
    #         for j in nodes:
    #             submission[f"{name}_{i}_{j}"] = int(A.loc[i, j])

    # s = pd.Series(submission).reset_index()
    # s.columns = [id_column_name, prediction_column_name]
    # return s


# %% [markdown]
# ### Local Training & Evaluation

# %%


# %% [markdown]
# ### CrunchDAO Test & Submit

# # %%
# X_train = pd.read_pickle("data/X_train.pickle")
# y_train = pd.read_pickle("data/y_train.pickle")
# print(f"Loaded {len(X_train)} training samples.")

# # %%
# train(X_train, y_train, model_directory_path="resources")

X_test = pd.read_pickle("data/X_test_reduced.pickle")
y_pred = infer(X_test, model_directory_path="resources",
              id_column_name="example_id", prediction_column_name="prediction")
# y_pred.to_parquet("prediction/v5.parquet")

# crunch.test(no_determinism_check=True)
