"""
v14_noderich.py — v11 + Enriched Node Head + CI Features

Two targeted changes over v11 (v8b + structural attention bias):

1. RICHER NODE HEAD:
   Old: gather 4 edges → MergeOperator(4×d → d) → Linear(d → 8)
   New: gather 4 edges → stack as (4, d) → self-attention over 4 edges
        → pairwise interaction features (products/differences)
        → merge [attended, interactions, ci_features] → MLP → 8

   WHY: Mediator requires jointly reasoning about X→v AND v→Y.
   The old MergeOperator just concatenates — no explicit pairwise signal.
   The new head lets v→X attend to X→v (direction evidence) and
   v→X attend to v→Y (chain evidence for Mediator).

2. CI FEATURES (10 per node):
   Computed in preprocessing, injected at node head via MLP.
   - CMI(v,X|Y), CMI(v,Y|X), CMI(X,Y|v)  — the 3 that define classes
   - MI(v,X), MI(v,Y)                     — marginal dependence
   - pcorr(v,X|rest), pcorr(v,Y|rest)     — conditional linear dependence
   - corr(v,X), corr(v,Y)                 — marginal linear
   - log(p)                               — graph dimension

Everything else (8ch conv, structural bias, edge head, shard training,
XY aug, DDP) is identical to v11.

Usage:
    # Base training
    python v14_noderich.py

    # With XY augmentation (use pre-built shards)
    python v14_noderich.py --xyaug --shard_dir dataset_cache/shards_v14/
"""

import typing
import os
import sys
import json
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree

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

# === Channel config (same as v8b) ===
BANDWIDTHS = [0.2, 0.5, 1.0]
N_CHANNELS = 2 + len(BANDWIDTHS) + len(BANDWIDTHS)  # 8

# === Structural bias (same as v11) ===
N_STRUCT_BIAS_TYPES = 6

# === CI features (NEW in v14) ===
N_NODE_CI_FEATURES = 10

# === Training ===
MAX_EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
AUG_NOISE_STD = 0.01

LOCAL_CACHE_DIR = "dataset_cache/"
CACHE_TAG = "v14"

IS_CLOUD_SUBMIT = False


# ============================================================
# Graph Utilities (unchanged)
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
# Feature Computation: Kernel Regression + ANM (from v8b)
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
# Feature Computation: Node-level CI Features (NEW in v14)
# ============================================================
def _knn_mi(x: np.ndarray, y: np.ndarray, k: int = 7) -> float:
    """KSG mutual information estimator."""
    n = len(x)
    if n < k + 5:
        return 0.0
    max_n = 500
    if n > max_n:
        idx = np.random.choice(n, max_n, replace=False)
        x, y = x[idx], y[idx]
        n = max_n
    x = ((x - x.mean()) / (x.std() + 1e-10))[:, None]
    y = ((y - y.mean()) / (y.std() + 1e-10))[:, None]
    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    dd, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = np.maximum(dd[:, -1], 1e-10)
    n_x = np.array([tree_x.query_ball_point(x[i], eps[i], p=np.inf, return_length=True) - 1
                     for i in range(n)])
    n_y = np.array([tree_y.query_ball_point(y[i], eps[i], p=np.inf, return_length=True) - 1
                     for i in range(n)])
    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)
    mi = digamma(k) + digamma(n) - np.mean(digamma(n_x) + digamma(n_y))
    return float(max(mi, 0.0))


def _knn_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = 7) -> float:
    """Frenzel-Pompe CMI estimator I(X;Y|Z)."""
    n = len(x)
    if n < k + 5:
        return 0.0
    max_n = 500
    if n > max_n:
        idx = np.random.choice(n, max_n, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        n = max_n
    x = ((x - x.mean()) / (x.std() + 1e-10))[:, None]
    y = ((y - y.mean()) / (y.std() + 1e-10))[:, None]
    z = ((z - z.mean()) / (z.std() + 1e-10))[:, None]
    xyz = np.hstack([x, y, z])
    xz = np.hstack([x, z])
    yz = np.hstack([y, z])
    tree_xyz = cKDTree(xyz)
    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z)
    dd, _ = tree_xyz.query(xyz, k=k + 1, p=np.inf)
    eps = np.maximum(dd[:, -1], 1e-10)
    n_xz = np.array([tree_xz.query_ball_point(xz[i], eps[i], p=np.inf, return_length=True) - 1
                      for i in range(n)])
    n_yz = np.array([tree_yz.query_ball_point(yz[i], eps[i], p=np.inf, return_length=True) - 1
                      for i in range(n)])
    n_z = np.array([tree_z.query_ball_point(z[i], eps[i], p=np.inf, return_length=True) - 1
                     for i in range(n)])
    n_xz = np.maximum(n_xz, 1)
    n_yz = np.maximum(n_yz, 1)
    n_z = np.maximum(n_z, 1)
    cmi = digamma(k) - np.mean(digamma(n_xz) + digamma(n_yz) - digamma(n_z))
    return float(max(cmi, 0.0))


def compute_node_ci_features(data: np.ndarray, cols: list) -> tuple:
    """
    Compute per-node CI features for all non-X/Y variables.

    Returns:
        node_features: (K, N_NODE_CI_FEATURES) float32
        node_names: list of K variable names
    """
    N, p = data.shape
    col_idx = {name: i for i, name in enumerate(cols)}
    x_idx, y_idx = col_idx["X"], col_idx["Y"]
    x_data, y_data = data[:, x_idx], data[:, y_idx]

    # Partial correlation matrix
    cov = np.cov(data.T)
    cov += 1e-6 * np.eye(p)
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        precision = np.eye(p)
    diag = np.sqrt(np.maximum(np.diag(precision), 1e-10))
    pcorr_matrix = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcorr_matrix, 0.0)

    non_xy = [c for c in cols if c not in ("X", "Y")]
    K = len(non_xy)
    features = np.zeros((K, N_NODE_CI_FEATURES), dtype=np.float32)

    for vi, v_name in enumerate(non_xy):
        v_i = col_idx[v_name]
        v_data = data[:, v_i]

        features[vi, 0] = _knn_cmi(v_data, x_data, y_data)  # CMI(v,X|Y)
        features[vi, 1] = _knn_cmi(v_data, y_data, x_data)  # CMI(v,Y|X)
        features[vi, 2] = _knn_cmi(x_data, y_data, v_data)  # CMI(X,Y|v)
        features[vi, 3] = _knn_mi(v_data, x_data)            # MI(v,X)
        features[vi, 4] = _knn_mi(v_data, y_data)            # MI(v,Y)
        features[vi, 5] = pcorr_matrix[v_i, x_idx]           # pcorr(v,X|rest)
        features[vi, 6] = pcorr_matrix[v_i, y_idx]           # pcorr(v,Y|rest)

        cx = np.corrcoef(v_data, x_data)[0, 1]
        cy = np.corrcoef(v_data, y_data)[0, 1]
        features[vi, 7] = cx if np.isfinite(cx) else 0.0     # corr(v,X)
        features[vi, 8] = cy if np.isfinite(cy) else 0.0     # corr(v,Y)
        features[vi, 9] = np.log(p)                           # log dim

    return features, non_xy


# ============================================================
# Structural Bias (matched exactly to v11)
# ============================================================
def build_struct_rel_matrix(p: int) -> np.ndarray:
    """
    Build structural relationship matrix between all edge pairs.

    Edge ordering: for u in range(p), for v in range(p), if u!=v.

    Types:
      0: Reverse pair     (u1=v2 AND v1=u2)
      1: Shared source    (u1=u2, not reverse)
      2: Shared target    (v1=v2, not reverse)
      3: Forward chain    (v1=u2, not reverse)
      4: Backward chain   (u1=v2, not reverse)
      5: Unrelated

    Returns: (E, E) int64 array where E = p*(p-1)
    """
    E: int = p * (p - 1)

    edge_uv: list = []
    for u in range(p):
        for v in range(p):
            if u != v:
                edge_uv.append((u, v))

    rel: np.ndarray = np.full((E, E), 5, dtype=np.int64)  # default: unrelated

    for e1 in range(E):
        u1, v1 = edge_uv[e1]
        for e2 in range(E):
            u2, v2 = edge_uv[e2]

            if u1 == v2 and v1 == u2:
                rel[e1, e2] = 0  # reverse pair
            elif u1 == u2:
                rel[e1, e2] = 1  # shared source
            elif v1 == v2:
                rel[e1, e2] = 2  # shared target
            elif v1 == u2:
                rel[e1, e2] = 3  # forward chain
            elif u1 == v2:
                rel[e1, e2] = 4  # backward chain
            # else: 5 (unrelated, already set)

    return rel


_STRUCT_REL_CACHE: dict = {}


def get_struct_rel_matrix(p: int) -> np.ndarray:
    global _STRUCT_REL_CACHE
    if p not in _STRUCT_REL_CACHE:
        _STRUCT_REL_CACHE[p] = build_struct_rel_matrix(p)
    return _STRUCT_REL_CACHE[p]


# ============================================================
# Data Preprocessing
# ============================================================
def build_edge_tensor(df: pd.DataFrame) -> tuple:
    """
    Build 8-channel edge tensor + node CI features.
    Returns: edge_data, edge_types, node_ci_features, node_names
    """
    cols = list(df.columns)
    p = len(cols)
    data = df.values.astype(np.float32)
    N = data.shape[0]

    # Kernel regression + ANM at each bandwidth
    coeff_maps, resid_maps = [], []
    for bw in BANDWIDTHS:
        cm, rm = compute_multivariate_kernel_coefficients(data, bandwidth=bw)
        coeff_maps.append(cm)
        resid_maps.append(rm)

    # Node CI features
    node_ci, node_names = compute_node_ci_features(data.astype(np.float64), cols)

    # Build edge tensors
    edges, edge_types_list = [], []
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

    edge_data = np.stack(edges, axis=0).astype(np.float32)
    edge_types = np.array(edge_types_list, dtype=np.int64)
    return edge_data, edge_types, node_ci, node_names


def _build_single(args):
    df, y_df = args
    edge_data, edge_types, node_ci, node_names = build_edge_tensor(df)
    cols = list(df.columns)
    p = len(cols)
    result = {
        "edge_data": edge_data, "edge_types": edge_types,
        "node_ci": node_ci, "node_names": node_names,
        "cols": cols, "p": p,
    }
    if y_df is not None:
        adj_np = y_df.values.astype(np.float32)
        adj_cols = list(y_df.columns)
        result["adj"] = adj_np; result["adj_cols"] = adj_cols
        edge_labels = []
        for i in range(p):
            for j in range(p):
                if i != j:
                    edge_labels.append(int(adj_np[adj_cols.index(cols[i]),
                                                  adj_cols.index(cols[j])]))
        result["edge_labels"] = np.array(edge_labels, dtype=np.int64)
        adj_df = pd.DataFrame(adj_np, index=adj_cols, columns=adj_cols)
        labels = get_labels(adj_df)
        result["node_labels"] = np.array([labels[v] for v in node_names], dtype=np.int64)
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
    max_K = max(item["node_ci"].shape[0] for item in batch)
    N = batch[0]["edge_data"].shape[2]
    C = batch[0]["edge_data"].shape[1]

    edge_data = torch.zeros(B, max_E, C, N)
    edge_types = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask = torch.zeros(B, max_E, dtype=torch.bool)
    node_ci = torch.zeros(B, max_K, N_NODE_CI_FEATURES)
    node_mask = torch.zeros(B, max_K, dtype=torch.bool)
    # Default to type 5 (unrelated) for padding — matches v11
    struct_rel = torch.full((B, max_E, max_E), 5, dtype=torch.long)

    has_labels = False
    edge_labels = torch.zeros(B, max_E, dtype=torch.long)
    node_labels = torch.zeros(B, max_K, dtype=torch.long)
    cols_list = []

    for b, item in enumerate(batch):
        E = item["edge_data"].shape[0]
        K = item["node_ci"].shape[0]
        edge_data[b, :E] = torch.from_numpy(item["edge_data"])
        edge_types[b, :E] = torch.from_numpy(item["edge_types"])
        edge_mask[b, :E] = True
        node_ci[b, :K] = torch.from_numpy(item["node_ci"])
        node_mask[b, :K] = True
        cols_list.append(item["cols"])

        # Build struct_rel from cache (not stored per item) — matches v11
        p = item["p"]
        rel_mat = get_struct_rel_matrix(p)
        struct_rel[b, :E, :E] = torch.from_numpy(rel_mat)

        if "edge_labels" in item:
            has_labels = True
            edge_labels[b, :E] = torch.from_numpy(item["edge_labels"])
            node_labels[b, :K] = torch.from_numpy(item["node_labels"])

    out = {"edge_data": edge_data, "edge_types": edge_types,
           "edge_mask": edge_mask, "node_ci": node_ci, "node_mask": node_mask,
           "struct_rel": struct_rel, "cols": cols_list,
           "_keys": [item.get("_key") for item in batch]}
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels
        out["node_mask_labels"] = node_mask
    return out


# ============================================================
# Model Components (shared with v8b/v11)
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
    """Self-attention with learned structural bias (matched to v11)."""
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
        nn.init.zeros_(self.struct_bias.weight)  # start neutral
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


# ============================================================
# NEW: Enriched Node Head
# ============================================================
class NodeCIProjector(nn.Module):
    """Project per-node CI scalars → d-dim."""
    def __init__(self, n_feat, d):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_feat, d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
    def forward(self, x):
        return self.proj(x)


class EnrichedNodeHead(nn.Module):
    """
    Richer node classification head.

    Old: MergeOperator([e_vx, e_vy, e_xv, e_yv]) → Linear(8)
    New:
      1. Stack 4 edges as (4, d) → 1-layer self-attention → (4, d)
      2. Pairwise interactions: 6 element-wise products between all pairs
         → Linear(6d → d)  (captures direction asymmetry, chain evidence, etc.)
      3. Pooled attended edges: mean of 4 attended edges → (d,)
      4. CI features: MLP(10 → d) → (d,)
      5. Merge [pooled, interactions, ci_emb] → MLP → 8 classes
    """
    def __init__(self, d=64, n_ci_features=N_NODE_CI_FEATURES):
        super().__init__()
        self.d = d

        # Self-attention over the 4 edges
        self.edge_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.edge_attn_norm = nn.LayerNorm(d)

        # Pairwise interaction projection
        # 6 pairs: (vx,vy), (vx,xv), (vx,yv), (vy,xv), (vy,yv), (xv,yv)
        self.interaction_proj = nn.Sequential(
            nn.Linear(6 * d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, d), nn.LayerNorm(d),
        )

        # CI features projector
        self.ci_proj = NodeCIProjector(n_ci_features, d)

        # Final merge: pooled + interactions + ci → classification
        self.merge = MergeOperator(n_inputs=3, d=d)
        self.classifier = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d, N_CLASSES),
        )

    def forward(self, e_vx, e_vy, e_xv, e_yv, ci_features):
        """
        e_vx, e_vy, e_xv, e_yv: (d,) each — gathered edge embeddings
        ci_features: (n_ci,) — CI scalar features for this node
        Returns: (N_CLASSES,) logits
        """
        # 1. Self-attention over 4 edges
        edges = torch.stack([e_vx, e_vy, e_xv, e_yv], dim=0).unsqueeze(0)  # (1, 4, d)
        attended, _ = self.edge_attn(edges, edges, edges)
        attended = self.edge_attn_norm(edges + attended)  # (1, 4, d)
        pooled = attended.squeeze(0).mean(dim=0)  # (d,)

        # 2. Pairwise interactions (element-wise product)
        interactions = torch.cat([
            e_vx * e_vy,   # both connect v to X/Y — confounder/collider signal
            e_vx * e_xv,   # v→X vs X→v — direction asymmetry for X
            e_vx * e_yv,   # v→X vs Y→v — cross signal
            e_vy * e_xv,   # v→Y vs X→v — cross signal
            e_vy * e_yv,   # v→Y vs Y→v — direction asymmetry for Y
            e_xv * e_yv,   # X→v vs Y→v — both point to v → collider signal
        ], dim=-1)  # (6d,)
        interaction_emb = self.interaction_proj(interactions)  # (d,)

        # 3. CI features
        ci_emb = self.ci_proj(ci_features)  # (d,)

        # 4. Merge + classify
        merged = self.merge([pooled, interaction_emb, ci_emb])
        return self.classifier(merged)


# ============================================================
# Full Model
# ============================================================
class ADIAModel(nn.Module):
    """
    v14: v8b channels + v11 structural bias + enriched node head + CI features.

    Pipeline:
      1. EdgeFeatureExtractor: (B,E,8,N) → (B,E,d)
      2. Merge [conv, type_emb] → (B,E,d)
      3. 2× SelfAttentionWithBias (structural bias from v11)
      4. Edge head: (B,E,d) → (B,E,2)
      5. EnrichedNodeHead: 4 edges + CI features → 8-class
    """
    def __init__(self, d=None, n_edge_types=None, aug_noise_std=AUG_NOISE_STD):
        super().__init__()
        d = d or D_MODEL
        n_edge_types = n_edge_types or N_EDGE_TYPES
        self.d = d
        self.aug_noise_std = aug_noise_std

        self.extractor = EdgeFeatureExtractor(d, n_blocks=5)
        self.edge_type_emb = nn.Embedding(n_edge_types, d)
        self.edge_merge = MergeOperator(n_inputs=2, d=d)
        self.attn_layers = nn.ModuleList([
            SelfAttentionWithBias(d, n_heads=4) for _ in range(2)
        ])
        self.edge_head = nn.Linear(d, 2)
        self.node_head = EnrichedNodeHead(d, N_NODE_CI_FEATURES)

    def forward(self, edge_data, edge_types, edge_mask, cols_list,
                node_ci=None, struct_rel=None):
        B, E, C, N = edge_data.shape

        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std

        # Edge feature extraction + merge
        edge_emb = self.extractor(edge_data.view(B*E, C, N)).view(B, E, self.d)
        type_emb = self.edge_type_emb(edge_types)
        edge_emb = self.edge_merge([edge_emb, type_emb])

        # Self-attention with structural bias
        pad_mask = ~edge_mask
        for attn in self.attn_layers:
            edge_emb = attn(edge_emb, struct_rel=struct_rel, key_padding_mask=pad_mask)

        # Edge head
        edge_logits = self.edge_head(edge_emb)

        # Node head with enriched processing
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
            for vi, node_name in enumerate(other_nodes):
                u = col_idx[node_name]
                e_vx = embs[edge_order[(u, x_idx)]]
                e_vy = embs[edge_order[(u, y_idx)]]
                e_xv = embs[edge_order[(x_idx, u)]]
                e_yv = embs[edge_order[(y_idx, u)]]

                # CI features for this node
                if node_ci is not None and vi < node_ci.shape[1]:
                    ci = node_ci[b, vi]
                else:
                    ci = torch.zeros(N_NODE_CI_FEATURES, device=embs.device)

                logits = self.node_head(e_vx, e_vy, e_xv, e_yv, ci)
                node_logits.append(logits)

            node_logits_list.append(
                torch.stack(node_logits) if node_logits else None
            )

        return edge_logits, node_logits_list


# ============================================================
# Lightning Module
# ============================================================
class ADIALightningModule(pl.LightningModule):
    def __init__(self, class_weights=None, lr=LR):
        super().__init__()
        self.model = ADIAModel()
        self.lr = lr
        if class_weights is not None:
            self.register_buffer("edge_w", torch.ones(2))
            self.register_buffer("node_w", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.register_buffer("edge_w", torch.ones(2))
            self.register_buffer("node_w", torch.ones(N_CLASSES))

    def forward(self, batch):
        return self.model(
            batch["edge_data"], batch["edge_types"], batch["edge_mask"],
            batch["cols"], node_ci=batch.get("node_ci"),
            struct_rel=batch.get("struct_rel"),
        )

    def _compute_loss(self, batch, prefix="train"):
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

        # Node loss
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
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-6)
        return [opt], [sched]


# ============================================================
# Data Loading
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
                result["_key"] = keys_list[idx]  # embed sample key
                items[idx] = result
            except Exception as e:
                print(f"Error at index {idx}: {e}")

    # Remove any None entries from failed workers
    items = [item for item in items if item is not None]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cached {len(items)} items to {cache_path}")
    return items


# ============================================================
# Inference
# ============================================================
def infer_batch_local(model, dataset, device="cuda"):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    all_preds = {}  # key: (sample_key, var_name) → class_index
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            _, node_logits_list = model(
                batch["edge_data"], batch["edge_types"], batch["edge_mask"],
                batch["cols"], node_ci=batch.get("node_ci"),
                struct_rel=batch.get("struct_rel"),
            )
            for b in range(len(node_logits_list)):
                nl = node_logits_list[b]
                if nl is None:
                    continue
                sample_key = batch["_keys"][b]
                cols = batch["cols"][b]
                non_xy = [c for c in cols if c not in ("X", "Y")]
                preds = nl.argmax(-1).cpu().numpy()
                for vi, vn in enumerate(non_xy):
                    all_preds[(sample_key, vn)] = int(preds[vi])
    return all_preds


# ============================================================
# Batch inference → adjacency DataFrames (for CrunchDAO submission)
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
def infer_batch_adj_local(dfs, model, device="cpu", batch_size=32, cache_dir=None):
    """
    Batch inference returning adjacency DataFrames.
    Used for CrunchDAO submission format.
    """
    model = model.eval()

    # Build items (with caching)
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"infer_items_{CACHE_TAG}_nk{N_KERNEL}.pkl")

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached infer items from {cache_path}...")
        with open(cache_path, "rb") as f:
            all_items = pickle.load(f)
    else:
        print(f"Building {len(dfs)} items for inference...")
        all_items = []
        ctx = mp.get_context('fork')
        args_list = [(df, None) for df in dfs]
        with ProcessPoolExecutor(max_workers=16, mp_context=ctx) as pool:
            futures = {pool.submit(_build_single, a): i for i, a in enumerate(args_list)}
            ordered = [None] * len(args_list)
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

    # Batched inference
    results = [None] * len(all_items)
    dataset = CausalEdgeDataset(all_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    idx_off = 0
    for batch in tqdm(loader, desc="Inferring"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        edge_logits, node_logits_list = model(
            batch["edge_data"], batch["edge_types"], batch["edge_mask"],
            batch["cols"], node_ci=batch.get("node_ci"),
            struct_rel=batch.get("struct_rel"),
        )

        for b in range(edge_logits.shape[0]):
            item = all_items[idx_off + b]
            cols = item["cols"]
            p = item["p"]

            # Build adjacency from edge head
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
# CrunchDAO API: train() and infer()
# ============================================================
def train(
    X_train: typing.Dict[str, pd.DataFrame],
    y_train: typing.Dict[str, pd.DataFrame],
    model_directory_path: str,
) -> None:
    """CrunchDAO train entry point."""
    keys = list(X_train.keys())

    # Build dataset
    cache_path = os.path.join(LOCAL_CACHE_DIR, f"train_dataset_{CACHE_TAG}_nk{N_KERNEL}.pkl")
    all_items = build_dataset(X_train, y_train, cache_path=cache_path, n_workers=16)

    # Class weights
    all_labels = []
    for item in all_items:
        if "node_labels" in item:
            all_labels.extend(item["node_labels"].tolist())
    label_counts = np.bincount(all_labels, minlength=N_CLASSES).astype(np.float32)
    class_weights = 1.0 / (label_counts + 1.0)
    class_weights = class_weights / class_weights.sum() * N_CLASSES

    # Train
    train_ds = CausalEdgeDataset(all_items)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    model = ADIALightningModule(class_weights=class_weights, lr=LR)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader)

    # Save
    state_dict = {k.replace("model.", ""): v
                  for k, v in model.state_dict().items() if k.startswith("model.")}
    os.makedirs(model_directory_path, exist_ok=True)
    path = os.path.join(model_directory_path, "model.pt")
    torch.save(state_dict, path)
    print(f"Model saved to {path}")


def infer(
    X_test: typing.Dict[str, pd.DataFrame],
    model_directory_path: str,
    id_column_name: str,
    prediction_column_name: str,
) -> pd.DataFrame:
    """CrunchDAO infer entry point."""
    path = os.path.join(model_directory_path, "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(aug_noise_std=0.0)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()

    names = list(X_test.keys())
    dfs = [X_test[n] for n in names]

    cache_dir = None if IS_CLOUD_SUBMIT else LOCAL_CACHE_DIR
    adj_list = infer_batch_adj_local(dfs, model, device=device, batch_size=64,
                                     cache_dir=cache_dir)

    submission = {}
    for name, A in zip(names, adj_list):
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
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    print("Loading data...")
    X_train = pd.read_pickle(os.path.join(args.data_dir, "X_train.pickle"))
    y_train = pd.read_pickle(os.path.join(args.data_dir, "y_train.pickle"))
    X_test = pd.read_pickle(os.path.join(args.data_dir, "X_test_reduced.pickle"))

    cache_path = os.path.join(args.cache_dir, f"train_dataset_{CACHE_TAG}_nk{N_KERNEL}.pkl")
    all_items = build_dataset(X_train, y_train, cache_path=cache_path, n_workers=args.n_workers)
    print(f"Training on all {len(all_items)} samples for {args.epochs} epochs")

    train_ds = CausalEdgeDataset(all_items)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    # Class weights
    all_labels = []
    for item in all_items:
        if "node_labels" in item:
            all_labels.extend(item["node_labels"].tolist())
    label_counts = np.bincount(all_labels, minlength=N_CLASSES).astype(np.float32)
    class_weights = 1.0 / (label_counts + 1.0)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.round(3)))}")

    model = ADIALightningModule(class_weights=class_weights, lr=args.lr)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision="32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, ckpt_path="/data/anhld48/phatnt/Graduation-Thesis-BsC-2026/src/lightning_logs/version_45/checkpoints/epoch=3-step=2940.ckpt")

    # Save final model
    state_dict = {k.replace("model.", ""): v
                  for k, v in model.state_dict().items() if k.startswith("model.")}
    os.makedirs("resources", exist_ok=True)
    torch.save(state_dict, "resources/model.pt")
    print("Saved final model to resources/model.pt")

    # Local eval
    y_test_path = os.path.join(args.data_dir, "y_test_reduced.pickle")
    if os.path.exists(y_test_path):
        print("\n=== Local Evaluation ===")
        test_cache = os.path.join(args.cache_dir, f"test_dataset_{CACHE_TAG}_nk{N_KERNEL}.pkl")
        test_items = build_dataset(X_test, None, cache_path=test_cache, n_workers=args.n_workers)
        test_ds = CausalEdgeDataset(test_items)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_model = ADIAModel()
        eval_model.load_state_dict(torch.load("resources/model.pt", map_location=device))
        eval_model.to(device)

        predictions = infer_batch_local(eval_model, test_ds, device=device)

        y_test = pd.read_pickle(y_test_path)
        correct, total = 0, 0
        per_class_c = np.zeros(N_CLASSES)
        per_class_t = np.zeros(N_CLASSES)

        test_keys = list(X_test.keys())
        for key in test_keys:
            y_df = y_test[key]
            adj_df = pd.DataFrame(y_df.values, index=list(y_df.columns), columns=list(y_df.columns))
            labels = get_labels(adj_df)
            cols = list(X_test[key].columns)
            for vn in [c for c in cols if c not in ("X", "Y")]:
                pred = predictions.get((key, vn))
                true = labels.get(vn)
                if pred is not None and true is not None:
                    total += 1
                    per_class_t[true] += 1
                    if pred == true:
                        correct += 1
                        per_class_c[true] += 1

        print(f"\nOverall accuracy: {correct/total:.4f} ({correct}/{total})")
        for c in range(N_CLASSES):
            if per_class_t[c] > 0:
                print(f"  {CLASS_NAMES[c]:20s}: {per_class_c[c]/per_class_t[c]:.4f} "
                      f"({int(per_class_c[c])}/{int(per_class_t[c])})")


if __name__ == "__main__":
    rank = int(os.environ.get("LOCAL_RANK", 0))
    main()