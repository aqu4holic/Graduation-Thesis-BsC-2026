"""
v15_multitower.py — Multi-Tower Architecture for Causal Role Classification

Built on v11 backbone (NOT v14 — v14's enriched head regressed).

Tower 1 (from v11, pretrained):
  8-channel conv1d → edge embeddings → self-attention with structural bias

Tower 2 (NEW):
  Per-edge scalar statistics → MLP → edge embeddings

Fused Node Head (simple, v11-style):
  Per-edge fusion: concat [t1, t2] → Linear(2d→d)
  Then: MergeOperator(4 fused edges) → Linear(d→8)

NO self-attention over edges, NO pairwise products, NO CI features.

Usage:
    python v15_multitower.py --pretrained resources/model_v11.pt
"""

import typing
import os
import sys
import json
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pytorch_lightning as pl

import networkx as nx
from sklearn.model_selection import train_test_split

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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

# === Channel config (same as v8b/v14) ===
BANDWIDTHS = [0.2, 0.5, 1.0]
N_CHANNELS = 2 + len(BANDWIDTHS) + len(BANDWIDTHS)  # 8

# === Structural bias (same as v11) ===
N_STRUCT_BIAS_TYPES = 6

# === Edge scalar stats (NEW in v15) ===
# 5 computed from raw data + 15 summarized from existing channels = 20
N_EDGE_STATS = 20

# === Training ===
MAX_EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
TOWER1_LR_SCALE = 0.1  # Tower 1 gets lr * 0.1
AUG_NOISE_STD = 0.01

LOCAL_CACHE_DIR = "dataset_cache/"
CACHE_TAG = "v15"

IS_CLOUD_SUBMIT = False


# ============================================================
# Graph Utilities (unchanged from v14)
# ============================================================
def graph_nodes_representation(graph, nodelist):
    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=nodelist).todense()
    return tuple(adjacency_matrix.flatten())


def create_graph_label():
    graphs = [
        (nx.DiGraph([("X", "Y"), ("v", "X"), ("v", "Y")]), "Confounder"),
        (nx.DiGraph([("X", "Y"), ("X", "v"), ("Y", "v")]), "Collider"),
        (nx.DiGraph([("X", "Y"), ("X", "v"), ("v", "Y")]), "Mediator"),
        (nx.DiGraph([("X", "Y"), ("v", "X")]),              "Cause of X"),
        (nx.DiGraph([("X", "Y"), ("v", "Y")]),              "Cause of Y"),
        (nx.DiGraph([("X", "Y"), ("X", "v")]),              "Consequence of X"),
        (nx.DiGraph([("X", "Y"), ("Y", "v")]),              "Consequence of Y"),
    ]
    g_indep = nx.DiGraph([("X", "Y")])
    g_indep.add_node("v")
    graphs.append((g_indep, "Independent"))

    adjacency_label = {}
    for G, label in graphs:
        key = graph_nodes_representation(G, ["v", "X", "Y"])
        adjacency_label[key] = CLASS_NAMES.index(label)
    return adjacency_label


_ADJACENCY_LABEL = create_graph_label()


def get_labels(adjacency_matrix, adjacency_label=None):
    if adjacency_label is None:
        adjacency_label = _ADJACENCY_LABEL
    result = {}
    for variable in adjacency_matrix.columns.drop(["X", "Y"]):
        submatrix = adjacency_matrix.loc[
            [variable, "X", "Y"], [variable, "X", "Y"]
        ]
        key = tuple(submatrix.values.flatten())
        result[variable] = adjacency_label.get(key, 7)
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
# Edge type encoding (unchanged)
# ============================================================
def _edge_type(u_name: str, v_name: str) -> int:
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
# Feature Computation: Kernel Regression + ANM (from v8b/v14)
# ============================================================
def compute_multivariate_kernel_coefficients(
    data: np.ndarray, n_sub: int = None, bandwidth: float = 0.5,
) -> tuple:
    """Returns (coeff_map, resid_map)."""
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

    resid_map = {}
    for j in range(p):
        c_all, other_cols = all_coeffs[j]
        c_nn = c_all[nearest]
        X_full = np.concatenate([np.ones((N, 1)),
                                 data[:, [k for k in range(p) if k != j]]], axis=1)
        y_hat = (c_nn * X_full).sum(axis=1)
        resid_map[j] = (data[:, j] - y_hat).astype(np.float32)

    return coeff_map, resid_map


# ============================================================
# NEW in v15: Per-Edge Scalar Statistics
# ============================================================
def compute_edge_scalar_stats(
    data: np.ndarray, cols: list,
    coeff_maps: list, resid_maps: list,
) -> np.ndarray:
    """
    Compute per-edge scalar features for Tower 2.

    Returns: (E, N_EDGE_STATS) float32  where E = p*(p-1)

    Features per edge (i→j):
      [0]  partial_corr(i,j)     — from precision matrix
      [1]  pearson(i,j)
      [2]  spearman(i,j)
      [3]  R²_forward (i→j)     — linear R² predicting j from i
      [4]  R²_reverse (j→i)     — linear R² predicting i from j
      --- Curve summary stats (from existing kernel/ANM channels) ---
      [5-7]   kernel_coeff_mean at bw 0.2, 0.5, 1.0
      [8-10]  kernel_coeff_std at bw 0.2, 0.5, 1.0
      [11-13] anm_resid_mean at bw 0.2, 0.5, 1.0
      [14-16] anm_resid_std at bw 0.2, 0.5, 1.0
      [17-19] anm_resid_skew at bw 0.2, 0.5, 1.0
    """
    N, p = data.shape

    # Partial correlation via precision matrix
    cov = np.cov(data.T)
    cov += 1e-6 * np.eye(p)
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        precision = np.eye(p)
    diag = np.sqrt(np.maximum(np.diag(precision), 1e-10))
    pcorr = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcorr, 0.0)

    edge_stats: list = []

    for i in range(p):
        xi = data[:, i]
        for j in range(p):
            if i == j:
                continue
            xj = data[:, j]
            stats = np.zeros(N_EDGE_STATS, dtype=np.float32)

            # [0] Partial correlation
            stats[0] = pcorr[i, j]

            # [1] Pearson
            if np.var(xi) > 1e-10 and np.var(xj) > 1e-10:
                pc = np.corrcoef(xi, xj)[0, 1]
                stats[1] = pc if np.isfinite(pc) else 0.0

            # [2] Spearman
            if np.var(xi) > 1e-10 and np.var(xj) > 1e-10:
                sc, _ = sp_stats.spearmanr(xi, xj)
                stats[2] = sc if np.isfinite(sc) else 0.0

            # [3] R² forward (i predicts j)
            if np.var(xi) > 1e-10:
                beta = np.cov(xi, xj)[0, 1] / np.var(xi)
                resid = xj - beta * xi
                ss_res = np.sum(resid ** 2)
                ss_tot = np.sum((xj - xj.mean()) ** 2) + 1e-10
                stats[3] = max(1.0 - ss_res / ss_tot, 0.0)

            # [4] R² reverse (j predicts i)
            if np.var(xj) > 1e-10:
                beta = np.cov(xi, xj)[0, 1] / np.var(xj)
                resid = xi - beta * xj
                ss_res = np.sum(resid ** 2)
                ss_tot = np.sum((xi - xi.mean()) ** 2) + 1e-10
                stats[4] = max(1.0 - ss_res / ss_tot, 0.0)

            # [5-19] Curve summary stats from kernel coeffs and ANM residuals
            for bw_idx in range(len(BANDWIDTHS)):
                # Kernel coefficient for edge (i→j) at this bandwidth
                kc = coeff_maps[bw_idx].get((i, j))
                if kc is not None:
                    stats[5 + bw_idx] = np.mean(kc)
                    stats[8 + bw_idx] = np.std(kc)

                # ANM residual for target j at this bandwidth
                ar = resid_maps[bw_idx].get(j)
                if ar is not None:
                    stats[11 + bw_idx] = np.mean(ar)
                    stats[14 + bw_idx] = np.std(ar)
                    # Skew — use scipy for safety, clip to [-10, 10]
                    sk = sp_stats.skew(ar)
                    stats[17 + bw_idx] = np.clip(sk, -10.0, 10.0) if np.isfinite(sk) else 0.0

            edge_stats.append(stats)

    result = np.stack(edge_stats, axis=0).astype(np.float32)  # (E, N_EDGE_STATS)
    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(result, -50.0, 50.0, out=result)  # hard clamp all features
    return result


# ============================================================
# Structural Bias (from v11/v14)
# ============================================================
def build_struct_rel_matrix(p: int) -> np.ndarray:
    E: int = p * (p - 1)
    edge_uv: list = []
    for u in range(p):
        for v in range(p):
            if u != v:
                edge_uv.append((u, v))
    rel: np.ndarray = np.full((E, E), 5, dtype=np.int64)
    for e1 in range(E):
        u1, v1 = edge_uv[e1]
        for e2 in range(E):
            u2, v2 = edge_uv[e2]
            if u1 == v2 and v1 == u2:
                rel[e1, e2] = 0
            elif u1 == u2:
                rel[e1, e2] = 1
            elif v1 == v2:
                rel[e1, e2] = 2
            elif v1 == u2:
                rel[e1, e2] = 3
            elif u1 == v2:
                rel[e1, e2] = 4
    return rel


_STRUCT_REL_CACHE: dict = {}


def get_struct_rel_matrix(p: int) -> np.ndarray:
    global _STRUCT_REL_CACHE
    if p not in _STRUCT_REL_CACHE:
        _STRUCT_REL_CACHE[p] = build_struct_rel_matrix(p)
    return _STRUCT_REL_CACHE[p]


# ============================================================
# Data Preprocessing (extended for v15)
# ============================================================
def build_edge_tensor(df: pd.DataFrame) -> tuple:
    """
    Build edge tensors + scalar stats (no CI features in v15).

    Returns: edge_data, edge_types, edge_stats
    """
    cols = list(df.columns)
    p = len(cols)
    data = df.values.astype(np.float32)
    N = data.shape[0]

    # Kernel regression + ANM at each bandwidth
    coeff_maps: list = []
    resid_maps: list = []
    for bw in BANDWIDTHS:
        cm, rm = compute_multivariate_kernel_coefficients(data, bandwidth=bw)
        coeff_maps.append(cm)
        resid_maps.append(rm)

    # Per-edge scalar statistics for Tower 2
    edge_stats = compute_edge_scalar_stats(data, cols, coeff_maps, resid_maps)

    # Build edge tensors (same as v11/v8b)
    edges: list = []
    edge_types_list: list = []
    for i, u_name in enumerate(cols):
        sort_idx = np.argsort(data[:, i])
        u_sorted = data[sort_idx, i]
        for j, v_name in enumerate(cols):
            if i == j:
                continue
            v_sorted_by_u = data[sort_idx, j]
            channels = [u_sorted, v_sorted_by_u]
            for cm in coeff_maps:
                channels.append(cm[(i, j)][sort_idx])
            for rm in resid_maps:
                channels.append(rm[j][sort_idx])
            edges.append(np.stack(channels, axis=0))
            edge_types_list.append(_edge_type(u_name, v_name))

    edge_data: np.ndarray = np.stack(edges, axis=0).astype(np.float32)
    edge_types: np.ndarray = np.array(edge_types_list, dtype=np.int64)
    return edge_data, edge_types, edge_stats


def _build_single(args):
    df, y_df = args
    edge_data, edge_types, edge_stats = build_edge_tensor(df)
    cols = list(df.columns)
    p = len(cols)
    result = {
        "edge_data": edge_data, "edge_types": edge_types,
        "edge_stats": edge_stats,
        "cols": cols, "p": p,
    }
    if y_df is not None:
        adj_np = y_df.values.astype(np.float32)
        adj_cols = list(y_df.columns)
        result["adj"] = adj_np
        result["adj_cols"] = adj_cols
        edge_labels: list = []
        for i in range(p):
            for j in range(p):
                if i != j:
                    edge_labels.append(int(adj_np[adj_cols.index(cols[i]),
                                                  adj_cols.index(cols[j])]))
        result["edge_labels"] = np.array(edge_labels, dtype=np.int64)
        adj_df = pd.DataFrame(adj_np, index=adj_cols, columns=adj_cols)
        labels = get_labels(adj_df)
        non_xy = [c for c in cols if c not in ("X", "Y")]
        result["node_labels"] = np.array([labels[v] for v in non_xy], dtype=np.int64)
    return result


# ============================================================
# Dataset & Collate
# ============================================================
class CausalEdgeDataset(Dataset):
    def __init__(self, items: list):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    B = len(batch)
    max_E = max(item["edge_data"].shape[0] for item in batch)
    max_K = max(len([c for c in item["cols"] if c not in ("X", "Y")]) for item in batch)
    N = batch[0]["edge_data"].shape[2]
    C = batch[0]["edge_data"].shape[1]

    edge_data = torch.zeros(B, max_E, C, N)
    edge_types = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask = torch.zeros(B, max_E, dtype=torch.bool)
    edge_stats = torch.zeros(B, max_E, N_EDGE_STATS)
    struct_rel = torch.full((B, max_E, max_E), 5, dtype=torch.long)

    has_labels = False
    edge_labels = torch.zeros(B, max_E, dtype=torch.long)
    node_labels = torch.zeros(B, max_K, dtype=torch.long)
    cols_list: list = []

    for b, item in enumerate(batch):
        E = item["edge_data"].shape[0]
        K = len([c for c in item["cols"] if c not in ("X", "Y")])
        edge_data[b, :E] = torch.from_numpy(item["edge_data"])
        edge_types[b, :E] = torch.from_numpy(item["edge_types"])
        edge_mask[b, :E] = True
        edge_stats[b, :E] = torch.from_numpy(item["edge_stats"])
        cols_list.append(item["cols"])

        p = item["p"]
        rel_mat = get_struct_rel_matrix(p)
        struct_rel[b, :E, :E] = torch.from_numpy(rel_mat)

        if "edge_labels" in item:
            has_labels = True
            edge_labels[b, :E] = torch.from_numpy(item["edge_labels"])
            node_labels[b, :K] = torch.from_numpy(item["node_labels"])

    out = {
        "edge_data": edge_data, "edge_types": edge_types,
        "edge_mask": edge_mask, "edge_stats": edge_stats,
        "struct_rel": struct_rel, "cols": cols_list,
        "_keys": [item.get("_key") for item in batch],
    }
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels
    return out


# ============================================================
# Model Components (shared with v14)
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
        self.linear = nn.Linear(n_channels or N_CHANNELS, d)
    def forward(self, x):
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class EdgeFeatureExtractor(nn.Module):
    def __init__(self, d=64, n_blocks=5):
        super().__init__()
        self.stem = StemLayer(d)
        self.blocks = nn.Sequential(*[ConvBlock(d) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x)
        return self.pool(x).squeeze(-1)


class SelfAttentionWithBias(nn.Module):
    def __init__(self, d=64, n_heads=4, n_bias_types=N_STRUCT_BIAS_TYPES):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.struct_bias = nn.Embedding(n_bias_types, n_heads)
        nn.init.zeros_(self.struct_bias.weight)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, struct_rel=None, key_padding_mask=None):
        B, E, _ = x.shape
        h, hd = self.n_heads, self.head_dim
        Q = self.q_proj(x).view(B, E, h, hd).transpose(1, 2)
        K = self.k_proj(x).view(B, E, h, hd).transpose(1, 2)
        V = self.v_proj(x).view(B, E, h, hd).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if struct_rel is not None:
            bias = self.struct_bias(struct_rel).permute(0, 3, 1, 2)
            attn_scores = attn_scores + bias
        if key_padding_mask is not None:
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, E, self.d)
        attn_out = self.out_proj(attn_out)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class NodeCIProjector(nn.Module):
    """Kept as placeholder — not used in v15 base, available for ablation."""
    pass


# ============================================================
# Tower 2 — Scalar Statistics Tower
# ============================================================
class ScalarStatTower(nn.Module):
    """
    Tower 2: Projects per-edge scalar statistics → edge embeddings.

    Input:  (B, E, N_EDGE_STATS) — 20 scalar features per edge
    Output: (B, E, d) — edge embeddings from scalar signal
    """
    def __init__(self, n_stats: int = N_EDGE_STATS, d: int = D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_stats, 4 * d),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )

    def forward(self, edge_stats: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_stats)


# ============================================================
# Simple Fused Node Head (v11-style, no attention/products)
# ============================================================
class FusedNodeHead(nn.Module):
    """
    Simple node classification head combining Tower 1 + Tower 2.
    Mirrors v11's approach: just merge edge embeddings → classify.

    v11 node head: MergeOperator(4 edges) → Linear(d→8)
    v15 node head: MergeOperator(4 fused edges) → Linear(d→8)

    Per-edge fusion: Linear(2d→d) on concat [t1_edge, t2_edge]
    Then: standard 4-input merge exactly like v11.
    """
    def __init__(self, d: int = D_MODEL):
        super().__init__()
        self.d = d

        # Per-edge fusion: concat Tower 1 + Tower 2 → d
        self.edge_fuse = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        # Same as v11: 4-input merge → classify
        self.node_merge = MergeOperator(n_inputs=4, d=d)
        self.node_head = nn.Linear(d, N_CLASSES)

    def forward(
        self,
        t1_vx: torch.Tensor, t1_vy: torch.Tensor,
        t1_xv: torch.Tensor, t1_yv: torch.Tensor,
        t2_vx: torch.Tensor, t2_vy: torch.Tensor,
        t2_xv: torch.Tensor, t2_yv: torch.Tensor,
    ) -> torch.Tensor:
        """All tensors are (d,). Returns (N_CLASSES,) logits."""
        # Fuse per edge position
        f_vx = self.edge_fuse(torch.cat([t1_vx, t2_vx], dim=-1))
        f_vy = self.edge_fuse(torch.cat([t1_vy, t2_vy], dim=-1))
        f_xv = self.edge_fuse(torch.cat([t1_xv, t2_xv], dim=-1))
        f_yv = self.edge_fuse(torch.cat([t1_yv, t2_yv], dim=-1))

        # Merge + classify (exactly like v11)
        merged = self.node_merge([f_vx, f_vy, f_xv, f_yv])
        return self.node_head(merged)


# ============================================================
# Full Multi-Tower Model
# ============================================================
class ADIAMultiTowerModel(nn.Module):
    """
    v15: v11 backbone (Tower 1) + scalar stat tower (Tower 2) + simple fusion.

    Tower 1 (pretrained from v11):
      EdgeFeatureExtractor → Merge[conv, type_emb] → 2× SelfAttentionWithBias
      → (B, E, d) conv edge embeddings

    Tower 2 (NEW, trained from scratch):
      ScalarStatTower: (B, E, N_EDGE_STATS) → (B, E, d) stat edge embeddings

    Edge head: from Tower 1 only (unchanged from v11)
    Node head: FusedNodeHead — fuse per edge, then v11-style merge → 8-class
    """
    def __init__(self, d: int = None, n_edge_types: int = None,
                 aug_noise_std: float = AUG_NOISE_STD):
        super().__init__()
        d = d or D_MODEL
        n_edge_types = n_edge_types or N_EDGE_TYPES
        self.d = d
        self.aug_noise_std = aug_noise_std

        # === Tower 1: Conv edge features (from v11) ===
        self.extractor = EdgeFeatureExtractor(d, n_blocks=5)
        self.edge_type_emb = nn.Embedding(n_edge_types, d)
        self.edge_merge = MergeOperator(n_inputs=2, d=d)
        self.attn_layers = nn.ModuleList([
            SelfAttentionWithBias(d, n_heads=4) for _ in range(2)
        ])

        # === Tower 2: Scalar statistics (NEW) ===
        self.stat_tower = ScalarStatTower(N_EDGE_STATS, d)

        # === Edge head (from Tower 1 only) ===
        self.edge_head = nn.Linear(d, 2)

        # === Simple Fused Node Head ===
        self.node_head = FusedNodeHead(d)

    def load_tower1_from_v11(self, v11_state_dict: dict) -> None:
        """Load Tower 1 weights from a v11 checkpoint."""
        tower1_prefixes = [
            "extractor.", "edge_type_emb.", "edge_merge.",
            "attn_layers.", "edge_head.",
        ]
        loaded: int = 0
        skipped: int = 0
        own_state = self.state_dict()

        for key, val in v11_state_dict.items():
            if any(key.startswith(pfx) for pfx in tower1_prefixes):
                if key in own_state and own_state[key].shape == val.shape:
                    own_state[key] = val
                    loaded += 1
                else:
                    skipped += 1
                    print(f"  [SKIP] {key} — shape mismatch or not in model")
            else:
                skipped += 1  # node_head/node_merge from v11 — don't load

        self.load_state_dict(own_state, strict=False)
        print(f"  Loaded {loaded} Tower 1 params from v11, skipped {skipped}")

    def tower1_parameters(self):
        """Parameters belonging to Tower 1 (for differential LR)."""
        t1_modules = [self.extractor, self.edge_type_emb,
                      self.edge_merge, self.attn_layers, self.edge_head]
        for m in t1_modules:
            yield from m.parameters()

    def tower2_and_head_parameters(self):
        """Parameters belonging to Tower 2 + FusedNodeHead (for higher LR)."""
        t2_modules = [self.stat_tower, self.node_head]
        for m in t2_modules:
            yield from m.parameters()

    def forward(
        self,
        edge_data: torch.Tensor, edge_types: torch.Tensor,
        edge_mask: torch.Tensor, cols_list: list,
        edge_stats: torch.Tensor = None,
        struct_rel: torch.Tensor = None,
    ):
        B, E, C, N = edge_data.shape

        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std

        # === Tower 1: Conv path (identical to v11) ===
        t1_emb = self.extractor(edge_data.view(B*E, C, N)).view(B, E, self.d)
        type_emb = self.edge_type_emb(edge_types)
        t1_emb = self.edge_merge([t1_emb, type_emb])

        pad_mask = ~edge_mask
        for attn in self.attn_layers:
            t1_emb = attn(t1_emb, struct_rel=struct_rel, key_padding_mask=pad_mask)

        # === Tower 2: Scalar stats path ===
        if edge_stats is not None:
            edge_stats = torch.nan_to_num(edge_stats, nan=0.0, posinf=0.0, neginf=0.0)
            t2_emb = self.stat_tower(edge_stats)
        else:
            t2_emb = torch.zeros_like(t1_emb)

        # === Edge head (Tower 1 only) ===
        edge_logits = self.edge_head(t1_emb)

        # === Fused Node Head ===
        node_logits_list: list = []
        for b in range(B):
            cols = cols_list[b]
            p = len(cols)
            col_idx = {name: i for i, name in enumerate(cols)}
            edge_order: dict = {}
            count = 0
            for ui in range(p):
                for vi in range(p):
                    if ui != vi:
                        edge_order[(ui, vi)] = count
                        count += 1

            x_idx, y_idx = col_idx.get("X"), col_idx.get("Y")
            other_nodes = [n for n in cols if n not in ("X", "Y")]

            if not other_nodes or x_idx is None or y_idx is None:
                node_logits_list.append(None)
                continue

            node_logits: list = []
            for vi, node_name in enumerate(other_nodes):
                u = col_idx[node_name]
                logits = self.node_head(
                    t1_emb[b, edge_order[(u, x_idx)]],
                    t1_emb[b, edge_order[(u, y_idx)]],
                    t1_emb[b, edge_order[(x_idx, u)]],
                    t1_emb[b, edge_order[(y_idx, u)]],
                    t2_emb[b, edge_order[(u, x_idx)]],
                    t2_emb[b, edge_order[(u, y_idx)]],
                    t2_emb[b, edge_order[(x_idx, u)]],
                    t2_emb[b, edge_order[(y_idx, u)]],
                )
                node_logits.append(logits)

            node_logits_list.append(
                torch.stack(node_logits) if node_logits else None
            )

        return edge_logits, node_logits_list


# ============================================================
# Lightning Module (updated for differential LR)
# ============================================================
class ADIALightningModule(pl.LightningModule):
    def __init__(self, class_weights=None, lr=LR, tower1_lr_scale=TOWER1_LR_SCALE,
                 tower1_pretrained=False, freeze_tower1_epochs=0):
        super().__init__()
        self.model = ADIAMultiTowerModel()
        self.lr = lr
        self.tower1_lr_scale = tower1_lr_scale
        self.tower1_pretrained = tower1_pretrained
        self.freeze_tower1_epochs = freeze_tower1_epochs  # freeze Tower 1 for N epochs
        self._tower1_frozen = False
        if class_weights is not None:
            self.register_buffer("edge_w", torch.ones(2))
            self.register_buffer("node_w", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.register_buffer("edge_w", torch.ones(2))
            self.register_buffer("node_w", torch.ones(N_CLASSES))

    def _freeze_tower1(self):
        for p in self.model.tower1_parameters():
            p.requires_grad = False
        self._tower1_frozen = True
        print("[v15] Tower 1 FROZEN")

    def _unfreeze_tower1(self):
        for p in self.model.tower1_parameters():
            p.requires_grad = True
        self._tower1_frozen = False
        print("[v15] Tower 1 UNFROZEN")

    def on_train_start(self):
        # Freeze Tower 1 at the beginning if requested
        if self.tower1_pretrained and self.freeze_tower1_epochs > 0:
            self._freeze_tower1()

    def on_train_epoch_start(self):
        # Unfreeze Tower 1 after freeze_tower1_epochs
        if (self._tower1_frozen
                and self.current_epoch >= self.freeze_tower1_epochs):
            self._unfreeze_tower1()

    def forward(self, batch):
        return self.model(
            batch["edge_data"], batch["edge_types"], batch["edge_mask"],
            batch["cols"],
            edge_stats=batch.get("edge_stats"),
            struct_rel=batch.get("struct_rel"),
        )

    def _compute_loss(self, batch, prefix="train"):
        edge_logits, node_logits_list = self(batch)
        B = len(node_logits_list)

        # Edge loss (Tower 1)
        edge_mask = batch["edge_mask"]
        edge_labels = batch["edge_labels"]
        valid = edge_mask.view(-1)
        edge_loss = F.cross_entropy(
            edge_logits.view(-1, 2)[valid],
            edge_labels.view(-1)[valid],
        )

        # Node loss (fused)
        all_nl, all_lab = [], []
        for b in range(B):
            nl = node_logits_list[b]
            if nl is None:
                continue
            K = nl.shape[0]
            all_nl.append(nl)
            all_lab.append(batch["node_labels"][b, :K])

        if all_nl:
            all_nl = torch.cat(all_nl)
            all_lab = torch.cat(all_lab)
            node_loss = F.cross_entropy(all_nl, all_lab, weight=self.node_w)

            if prefix == "val":
                preds = all_nl.argmax(-1)
                acc = (preds == all_lab).float().mean()
                self.log(f"{prefix}_acc", acc, prog_bar=True, batch_size=B)
                for c in range(N_CLASSES):
                    m = all_lab == c
                    if m.sum() > 0:
                        self.log(f"{prefix}_acc_{CLASS_NAMES[c]}",
                                 (preds[m] == c).float().mean(), batch_size=B)
        else:
            node_loss = torch.tensor(0.0, device=self.device)

        total = edge_loss + node_loss
        self.log(f"{prefix}_loss", total, prog_bar=True, batch_size=B)
        return total

    def training_step(self, batch, batch_idx):
        edge_logits, node_logits_list = self(batch)
        B = len(node_logits_list)

        # Edge loss
        edge_mask = batch["edge_mask"]
        edge_labels = batch["edge_labels"]
        valid = edge_mask.view(-1)
        edge_loss = F.cross_entropy(
            edge_logits.view(-1, 2)[valid],
            edge_labels.view(-1)[valid],
        )

        # Node loss + accuracy
        all_nl, all_lab = [], []
        for b in range(B):
            nl = node_logits_list[b]
            if nl is None:
                continue
            K = nl.shape[0]
            all_nl.append(nl)
            all_lab.append(batch["node_labels"][b, :K])

        if all_nl:
            all_nl = torch.cat(all_nl)
            all_lab = torch.cat(all_lab)
            node_loss = F.cross_entropy(all_nl, all_lab, weight=self.node_w)
            preds = all_nl.argmax(-1)
            acc = (preds == all_lab).float().mean()
            self.log("train_acc", acc, prog_bar=True, batch_size=B)

            # Log unique predictions to detect class collapse
            n_unique = len(preds.unique())
            self.log("train_n_unique_preds", float(n_unique), prog_bar=True, batch_size=B)
        else:
            node_loss = torch.tensor(0.0, device=self.device)

        total = edge_loss + node_loss
        self.log("train_loss", total, prog_bar=True, batch_size=B)
        self.log("train_edge_loss", edge_loss, batch_size=B)
        self.log("train_node_loss", node_loss, batch_size=B)
        return total

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        # Only use differential LR if Tower 1 was initialized from pretrained weights.
        # Training from scratch: both towers need full LR.
        t1_lr = self.lr * self.tower1_lr_scale if self.tower1_pretrained else self.lr
        opt = torch.optim.AdamW([
            {"params": list(self.model.tower1_parameters()), "lr": t1_lr},
            {"params": list(self.model.tower2_and_head_parameters()), "lr": self.lr},
        ], weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=MAX_EPOCHS, eta_min=1e-6,
        )
        return [opt], [sched]


# ============================================================
# Data Loading (from v14)
# ============================================================
def build_dataset(X_data, y_data, cache_path=None, n_workers=16):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            items = pickle.load(f)
        print(f"  Loaded {len(items)} items")
        return items

    keys_list = list(X_data.keys())
    print(f"Building dataset ({len(keys_list)} samples)...")
    args_list = [(X_data[k], y_data[k] if y_data else None) for k in keys_list]

    items = [None] * len(args_list)
    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_build_single, a): i for i, a in enumerate(args_list)}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[f]
            try:
                result = f.result()
                result["_key"] = keys_list[idx]
                items[idx] = result
            except Exception as e:
                print(f"Error at index {idx}: {e}")

    items = [item for item in items if item is not None]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cached {len(items)} items to {cache_path}")
    return items


# ============================================================
# Inference — one clean function, matches v11 pattern
# ============================================================
_CLASS_TO_EDGES = {
    "Confounder":        lambda n: [(n, "X"), (n, "Y")],
    "Collider":          lambda n: [("X", n), ("Y", n)],
    "Mediator":          lambda n: [("X", n), (n, "Y")],
    "Cause of X":        lambda n: [(n, "X")],
    "Cause of Y":        lambda n: [(n, "Y")],
    "Consequence of X":  lambda n: [("X", n)],
    "Consequence of Y":  lambda n: [("Y", n)],
    "Independent":       lambda n: [],
}


@torch.no_grad()
def infer_batch_local(dfs, model, device="cpu", batch_size=64, cache_dir=None):
    """
    Takes raw DataFrames → returns list of adjacency DataFrames.
    Single function: builds tensors, runs model, converts to adjacency.
    """
    model = model.eval()

    # Build or load cached items
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"infer_items_{CACHE_TAG}_nk{N_KERNEL}.pkl")

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached infer items from {cache_path}...")
        with open(cache_path, "rb") as f:
            all_items = pickle.load(f)
    else:
        print(f"Building {len(dfs)} items for inference...")
        ctx = mp.get_context('fork')
        n_workers = max(1, mp.cpu_count() - 1)
        args_list = [(df, None) for df in dfs]
        ordered = [None] * len(args_list)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_build_single, a): i for i, a in enumerate(args_list)}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Building"):
                idx = futures[f]
                try:
                    ordered[idx] = f.result()
                except Exception as e:
                    print(f"Error at {idx}: {e}")
        all_items = [x for x in ordered if x is not None]

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(all_items, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Run model
    dataset = CausalEdgeDataset(all_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    results: list = [None] * len(all_items)
    idx_off: int = 0

    for batch in tqdm(loader, desc="Inference"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        edge_logits, node_logits_list = model(
            batch["edge_data"], batch["edge_types"], batch["edge_mask"],
            batch["cols"],
            edge_stats=batch.get("edge_stats"),
            struct_rel=batch.get("struct_rel"),
        )

        for b in range(edge_logits.shape[0]):
            item = all_items[idx_off + b]
            cols = item["cols"]
            p = item["p"]

            # Build DAG from edge head
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

            # Override with node head predictions (more accurate)
            if node_logits_list[b] is not None:
                other_nodes = [n for n in cols if n not in ("X", "Y")]
                node_preds = torch.argmax(node_logits_list[b], dim=-1)
                for k_node, node_name in enumerate(other_nodes):
                    pred_class = CLASS_NAMES[node_preds[k_node].item()]
                    A.loc[node_name, :] = 0
                    A.loc[:, node_name] = 0
                    for (src, dst) in _CLASS_TO_EDGES[pred_class](node_name):
                        A.loc[src, dst] = 1

            results[idx_off + b] = A
        idx_off += edge_logits.shape[0]

    return results


# ============================================================
# CrunchDAO API
# ============================================================
def train(
    X_train: typing.Dict[str, pd.DataFrame],
    y_train: typing.Dict[str, pd.DataFrame],
    model_directory_path: str,
) -> None:
    cache_path = os.path.join(LOCAL_CACHE_DIR, f"train_dataset_{CACHE_TAG}_nk{N_KERNEL}.pkl")
    all_items = build_dataset(X_train, y_train, cache_path=cache_path, n_workers=16)

    all_labels: list = []
    for item in all_items:
        if "node_labels" in item:
            all_labels.extend(item["node_labels"].tolist())
    label_counts = np.bincount(all_labels, minlength=N_CLASSES).astype(np.float32)
    class_weights = 1.0 / (label_counts + 1.0)
    class_weights = class_weights / class_weights.sum() * N_CLASSES

    train_ds = CausalEdgeDataset(all_items)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    lightning_module = ADIALightningModule(class_weights=class_weights, lr=LR)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    trainer.fit(lightning_module, train_loader)

    state_dict = {k.replace("model.", ""): v
                  for k, v in lightning_module.state_dict().items() if k.startswith("model.")}
    os.makedirs(model_directory_path, exist_ok=True)
    torch.save(state_dict, os.path.join(model_directory_path, "model.pt"))


def infer(
    X_test: typing.Dict[str, pd.DataFrame],
    model_directory_path: str,
    id_column_name: str,
    prediction_column_name: str,
) -> pd.DataFrame:
    path = os.path.join(model_directory_path, "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAMultiTowerModel(aug_noise_std=0.0)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()

    dfs = [X_test[n] for n in X_test]
    cache_dir = None if IS_CLOUD_SUBMIT else LOCAL_CACHE_DIR
    adj_list = infer_batch_local(dfs, model, device=device, batch_size=64,
                                 cache_dir=cache_dir)

    submission: dict = {}
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
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--cache_dir", default=LOCAL_CACHE_DIR)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n_workers", type=int, default=os.cpu_count() - 1)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to v11 model checkpoint to initialize Tower 1")
    parser.add_argument("--tower1_lr_scale", type=float, default=TOWER1_LR_SCALE,
                        help="LR multiplier for Tower 1 (default: 0.1)")
    parser.add_argument("--freeze_tower1_epochs", type=int, default=15,
                        help="Freeze Tower 1 for this many epochs when using --pretrained (default: 15)")
    args = parser.parse_args()

    print("Loading data...")
    X_train = pd.read_pickle(os.path.join(args.data_dir, "X_train.pickle"))
    y_train = pd.read_pickle(os.path.join(args.data_dir, "y_train.pickle"))
    X_test = pd.read_pickle(os.path.join(args.data_dir, "X_test_reduced.pickle"))

    cache_path = os.path.join(args.cache_dir, f"train_dataset_{CACHE_TAG}_nk{N_KERNEL}.pkl")
    all_items = build_dataset(X_train, y_train, cache_path=cache_path, n_workers=args.n_workers)
    print(f"Training on {len(all_items)} samples for {args.epochs} epochs")

    train_ds = CausalEdgeDataset(all_items)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    # Class weights
    all_labels: list = []
    for item in all_items:
        if "node_labels" in item:
            all_labels.extend(item["node_labels"].tolist())
    label_counts = np.bincount(all_labels, minlength=N_CLASSES).astype(np.float32)
    class_weights = 1.0 / (label_counts + 1.0)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.round(3)))}")

    lightning_module = ADIALightningModule(
        class_weights=class_weights, lr=args.lr,
        tower1_lr_scale=args.tower1_lr_scale,
        tower1_pretrained=(args.pretrained is not None),
        freeze_tower1_epochs=args.freeze_tower1_epochs if args.pretrained else 0,
    )

    # Load pretrained Tower 1 from v11
    args.pretrained = args.pretrained or os.path.join("resources", "model_v11_structbias_xyaug.pt")
    if args.pretrained:
        print(f"\nLoading Tower 1 from v11 checkpoint: {args.pretrained}")
        v11_state = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        lightning_module.model.load_tower1_from_v11(v11_state)

    strategy = "auto"
    if args.devices > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)  # needed for Tower 1 freezing

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=strategy,
        precision="32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    trainer.fit(lightning_module, train_loader)

    # Only rank 0 saves and evaluates
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank != 0:
        return

    # Save final model — strip Lightning "model." prefix and DDP "module." prefix
    state_dict = {}
    for k, v in lightning_module.state_dict().items():
        if k.startswith("model."):
            clean_key = k[len("model."):]
            # DDP may add another module. prefix
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module."):]
            state_dict[clean_key] = v
    os.makedirs("resources", exist_ok=True)
    torch.save(state_dict, "resources/model.pt")
    print("Saved final model to resources/model.pt")

    # Count params
    t1_params = sum(p.numel() for p in lightning_module.model.tower1_parameters())
    t2_params = sum(p.numel() for p in lightning_module.model.tower2_and_head_parameters())
    print(f"Tower 1 params: {t1_params:,}  |  Tower 2 + Head params: {t2_params:,}")
    print(f"Total: {t1_params + t2_params:,}")

    # Local eval
    y_test_path = os.path.join(args.data_dir, "y_test_reduced.pickle")
    if os.path.exists(y_test_path):
        print("\n=== Local Evaluation ===")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_model = ADIAMultiTowerModel(aug_noise_std=0.0)
        eval_model.load_state_dict(torch.load("resources/model.pt", map_location=device))
        eval_model.to(device)

        dfs = [X_test[n] for n in X_test]
        adj_list = infer_batch_local(dfs, eval_model, device=device, batch_size=64,
                                     cache_dir=args.cache_dir)

        y_test = pd.read_pickle(y_test_path)
        test_keys = list(X_test.keys())
        correct, total = 0, 0
        per_class_c = np.zeros(N_CLASSES)
        per_class_t = np.zeros(N_CLASSES)

        for key, A_pred in zip(test_keys, adj_list):
            # Ground truth labels
            y_df = y_test[key]
            adj_true = pd.DataFrame(y_df.values, index=list(y_df.columns),
                                    columns=list(y_df.columns))
            true_labels = get_labels(adj_true)

            # Predicted labels (same path as evaluate.py)
            pred_labels = get_labels(A_pred)

            cols = list(X_test[key].columns)
            for vn in [c for c in cols if c not in ("X", "Y")]:
                true = true_labels.get(vn)
                pred = pred_labels.get(vn)
                if true is not None and pred is not None:
                    total += 1
                    per_class_t[true] += 1
                    if pred == true:
                        correct += 1
                        per_class_c[true] += 1

        print(f"\nOverall accuracy: {correct/total:.4f} ({correct}/{total})")
        bal_acc: float = 0.0
        n_classes_seen: int = 0
        for c in range(N_CLASSES):
            if per_class_t[c] > 0:
                ca = per_class_c[c] / per_class_t[c]
                bal_acc += ca
                n_classes_seen += 1
                print(f"  {CLASS_NAMES[c]:20s}: {ca:.4f} "
                      f"({int(per_class_c[c])}/{int(per_class_t[c])})")
        bal_acc /= max(n_classes_seen, 1)
        print(f"\nBalanced accuracy: {bal_acc:.4f}")


if __name__ == "__main__":
    main()
