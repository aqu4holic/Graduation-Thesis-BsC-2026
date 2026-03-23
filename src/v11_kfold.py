"""
v11_kfold.py — ADIA Causal Discovery
  v11 Structural Attention Bias — 5-Fold Cross-Validation

Validates v11 score reliability. Splits 23.5K into 5 folds,
trains on 4, evaluates on 1. Reports mean ± std across folds.

Usage:
    python v11_kfold.py
"""

# @crunch/keep:on
import crunch

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import typing
import gc
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pytorch_lightning as pl
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
N_CLASSES: int = 8
N_EDGE_TYPES: int = 7
D_MODEL: int = 64
N_HEADS: int = 4
N_STRUCT_TYPES: int = 6  # structural relationship types
CLASS_NAMES: list[str] = [
    "Confounder", "Collider", "Mediator",
    "Cause of X", "Cause of Y",
    "Consequence of X", "Consequence of Y",
    "Independent",
]

BANDWIDTHS: list[float] = [0.2, 0.5, 1.0]
N_CHANNELS: int = 2 + len(BANDWIDTHS) + len(BANDWIDTHS)  # 8

MAX_EPOCHS: int = 30
BATCH_SIZE: int = 16
LR: float = 1e-3
AUG_NOISE_STD: float = 0.01
LOCAL_CACHE_DIR: str = "dataset_cache/"
IS_CLOUD_SUBMIT: bool = False


# ============================================================
# Graph Utilities
# ============================================================
def graph_nodes_representation(
    graph: nx.DiGraph, nodelist: list[str]
) -> tuple:
    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=nodelist).todense()
    return tuple(adjacency_matrix.flatten())


def create_graph_label() -> tuple[dict, dict]:
    graph_label: dict[nx.DiGraph, str] = {
        nx.DiGraph([("X", "Y"), ("v", "X"), ("v", "Y")]): "Confounder",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("Y", "v")]): "Collider",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("v", "Y")]): "Mediator",
        nx.DiGraph([("X", "Y"), ("v", "X")]): "Cause of X",
        nx.DiGraph([("X", "Y"), ("v", "Y")]): "Cause of Y",
        nx.DiGraph([("X", "Y"), ("X", "v")]): "Consequence of X",
        nx.DiGraph([("X", "Y"), ("Y", "v")]): "Consequence of Y",
        nx.DiGraph({"X": ["Y"], "v": []}): "Independent",
    }
    nodelist: list[str] = ["v", "X", "Y"]
    adjacency_label: dict[tuple, str] = {
        graph_nodes_representation(graph, nodelist): label
        for graph, label in graph_label.items()
    }
    return graph_label, adjacency_label


_GRAPH_LABEL, _ADJACENCY_LABEL = None, None


def get_adjacency_label() -> dict[tuple, str]:
    global _GRAPH_LABEL, _ADJACENCY_LABEL
    if _ADJACENCY_LABEL is None:
        _GRAPH_LABEL, _ADJACENCY_LABEL = create_graph_label()
    return _ADJACENCY_LABEL


def get_labels(
    adjacency_matrix: pd.DataFrame, adjacency_label: dict[tuple, str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for variable in adjacency_matrix.columns.drop(["X", "Y"]):
        submatrix = adjacency_matrix.loc[
            [variable, "X", "Y"], [variable, "X", "Y"]
        ]
        key: tuple = tuple(submatrix.values.flatten())
        result[variable] = adjacency_label[key]
    return result


def transform_proba_to_DAG(
    nodes: list[str], pred: np.ndarray
) -> np.ndarray:
    G: nx.DiGraph = nx.DiGraph()
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
# Data Preprocessing (identical to v8b)
# ============================================================
def _edge_type(u_name: str, v_name: str) -> int:
    uX: bool = u_name == "X"
    uY: bool = u_name == "Y"
    vX: bool = v_name == "X"
    vY: bool = v_name == "Y"
    if uX and not vY:  return 0
    if uX and vY:      return 1
    if uY and not vX:  return 2
    if uY and vX:      return 3
    if not uX and not uY and vX: return 4
    if not uX and not uY and vY: return 5
    return 6


def compute_multivariate_kernel_coefficients(
    data: np.ndarray, sub_idx: np.ndarray, bandwidth: float = 0.5,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[int, np.ndarray]]:
    N: int
    p: int
    N, p = data.shape
    n_sub: int = len(sub_idx)
    data_sub: np.ndarray = data[sub_idx]

    diff: np.ndarray = data_sub[:, None, :] - data_sub[None, :, :]
    sq_dist: np.ndarray = (diff ** 2).sum(axis=-1)
    W: np.ndarray = np.exp(-sq_dist / (2 * bandwidth ** 2))

    all_coeffs: dict[int, tuple] = {}
    for j in range(p):
        other_cols: list[int] = [k for k in range(p) if k != j]
        X_design: np.ndarray = np.concatenate(
            [np.ones((n_sub, 1)), data_sub[:, other_cols]], axis=1
        )
        y_target: np.ndarray = data_sub[:, j]
        A_all: np.ndarray = np.einsum('il,la,lb->iab', W, X_design, X_design)
        b_all: np.ndarray = np.einsum('il,la,l->ia', W, X_design, y_target)
        reg: np.ndarray = 1e-6 * np.eye(p)[None, :, :]
        try:
            c_all: np.ndarray = np.linalg.solve(
                A_all + reg, b_all[:, :, None]
            ).squeeze(-1)
        except np.linalg.LinAlgError:
            c_all = np.zeros((n_sub, p))
        all_coeffs[j] = (c_all, other_cols)

    dist_to_sub: np.ndarray = np.sum(
        (data[:, None, :] - data_sub[None, :, :]) ** 2, axis=-1
    )
    nearest: np.ndarray = np.argmin(dist_to_sub, axis=1)

    coeff_map: dict[tuple[int, int], np.ndarray] = {}
    resid_map: dict[int, np.ndarray] = {}

    for j in range(p):
        c_all, other_cols = all_coeffs[j]
        for idx_in_other, k in enumerate(other_cols):
            coeff_sub: np.ndarray = c_all[:, idx_in_other + 1]
            coeff_map[(k, j)] = coeff_sub[nearest].astype(np.float32)
        c_nn: np.ndarray = c_all[nearest]
        X_full: np.ndarray = np.concatenate(
            [np.ones((N, 1)), data[:, other_cols]], axis=1
        )
        y_hat: np.ndarray = np.sum(c_nn * X_full, axis=1)
        resid_map[j] = (data[:, j] - y_hat).astype(np.float32)

    return coeff_map, resid_map


def build_edge_tensor(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    cols: list[str] = list(df.columns)
    p: int = len(cols)
    data: np.ndarray = df.values.astype(np.float32)
    N: int = data.shape[0]

    n_sub: int = min(N_KERNEL, N)
    sub_idx: np.ndarray = np.random.choice(N, n_sub, replace=False)

    coeff_maps: list[dict] = []
    resid_maps: list[dict] = []
    for bw in BANDWIDTHS:
        coeff_map, resid_map = compute_multivariate_kernel_coefficients(
            data, sub_idx, bandwidth=bw
        )
        coeff_maps.append(coeff_map)
        resid_maps.append(resid_map)

    edges: list[np.ndarray] = []
    edge_types: list[int] = []

    for i, u_name in enumerate(cols):
        sort_idx: np.ndarray = np.argsort(data[:, i])
        u_sorted: np.ndarray = data[sort_idx, i]

        for j, v_name in enumerate(cols):
            if i == j:
                continue
            v_sorted_by_u: np.ndarray = data[sort_idx, j]
            kernel_channels: list[np.ndarray] = [
                cm[(i, j)][sort_idx] for cm in coeff_maps
            ]
            anm_channels: list[np.ndarray] = [
                rm[j][sort_idx] for rm in resid_maps
            ]
            channels: list[np.ndarray] = (
                [u_sorted, v_sorted_by_u]
                + kernel_channels
                + anm_channels
            )
            edge_tensor: np.ndarray = np.stack(channels, axis=0)
            edges.append(edge_tensor)
            edge_types.append(_edge_type(u_name, v_name))

    edge_data: np.ndarray = np.stack(edges, axis=0).astype(np.float32)
    edge_types_arr: np.ndarray = np.array(edge_types, dtype=np.int64)
    return edge_data, edge_types_arr


# ============================================================
# Structural Relationship Matrix
# ============================================================
def build_struct_rel_matrix(p: int) -> np.ndarray:
    """Build E×E matrix of structural relationship types for a p-node graph.

    Edge ordering: for u in range(p), for v in range(p), if u!=v.
    Edge index: u*(p-1) + v - (1 if v > u else 0)

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

    # Build (u, v) for each edge index
    edge_uv: list[tuple[int, int]] = []
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


# Cache struct rel matrices per p (only 3-10 variables in this dataset)
_STRUCT_REL_CACHE: dict[int, np.ndarray] = {}


def get_struct_rel_matrix(p: int) -> np.ndarray:
    global _STRUCT_REL_CACHE
    if p not in _STRUCT_REL_CACHE:
        _STRUCT_REL_CACHE[p] = build_struct_rel_matrix(p)
    return _STRUCT_REL_CACHE[p]


# ============================================================
# Sample Builder
# ============================================================
def _build_single(args: tuple) -> dict:
    df: pd.DataFrame
    y_df: pd.DataFrame | None
    df, y_df = args

    edge_data: np.ndarray
    edge_types: np.ndarray
    edge_data, edge_types = build_edge_tensor(df)

    cols: list[str] = list(df.columns)
    p: int = len(cols)
    result: dict = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "cols": cols,
        "p": p,
    }

    if y_df is not None:
        adj_np: np.ndarray = y_df.values.astype(np.float32)
        adj_cols: list[str] = list(y_df.columns)
        result["adj"] = adj_np

        edge_labels: list[int] = []
        for ui in range(p):
            for vi in range(p):
                if ui != vi:
                    edge_labels.append(int(adj_np[ui, vi]))
        result["edge_labels"] = np.array(edge_labels, dtype=np.int64)

        _, adjacency_label = create_graph_label()
        df_adj: pd.DataFrame = pd.DataFrame(
            adj_np, columns=adj_cols, index=adj_cols
        )
        node_labels_dict: dict[str, str] = get_labels(df_adj, adjacency_label)
        other_nodes: list[str] = [c for c in cols if c not in ("X", "Y")]
        node_labels: list[int] = [
            CLASS_NAMES.index(node_labels_dict[n]) for n in other_nodes
        ]
        result["node_labels"] = np.array(node_labels, dtype=np.int64)
        result["other_nodes"] = other_nodes

    return result


# ============================================================
# Dataset & Collate
# ============================================================
class InMemoryDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples: list[dict] = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item: dict = self.samples[idx]
        result: dict = {
            "edge_data": torch.from_numpy(item["edge_data"]),
            "edge_types": torch.from_numpy(item["edge_types"]),
            "cols": item["cols"],
            "p": item["p"],
        }
        if "edge_labels" in item:
            result["edge_labels"] = torch.from_numpy(item["edge_labels"])
            result["node_labels"] = torch.from_numpy(item["node_labels"])
        return result


def collate_fn(batch: list[dict]) -> dict:
    max_E: int = max(item["edge_data"].shape[0] for item in batch)
    B: int = len(batch)

    edge_data: torch.Tensor = torch.zeros(B, max_E, N_CHANNELS, N_OBS)
    edge_types: torch.Tensor = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask: torch.Tensor = torch.zeros(B, max_E, dtype=torch.bool)

    # Structural relationship matrix: (B, max_E, max_E)
    # Default to type 5 (unrelated) for padding
    struct_rel: torch.Tensor = torch.full(
        (B, max_E, max_E), 5, dtype=torch.long
    )

    max_K: int = max(
        (item["node_labels"].shape[0] if "node_labels" in item else 0)
        for item in batch
    )
    edge_labels: torch.Tensor = torch.full((B, max_E), -1, dtype=torch.long)
    node_labels: torch.Tensor = torch.full((B, max_K), -1, dtype=torch.long)
    node_mask: torch.Tensor = torch.zeros(B, max_K, dtype=torch.bool)

    cols_list: list[list[str]] = []
    has_labels: bool = False

    for b, item in enumerate(batch):
        E: int = item["edge_data"].shape[0]
        edge_data[b, :E] = item["edge_data"]
        edge_types[b, :E] = item["edge_types"]
        edge_mask[b, :E] = True
        cols_list.append(item["cols"])

        # Build struct rel matrix for this graph's p
        p: int = item["p"]
        rel_mat: np.ndarray = get_struct_rel_matrix(p)
        struct_rel[b, :E, :E] = torch.from_numpy(rel_mat)

        if "edge_labels" in item:
            has_labels = True
            edge_labels[b, :E] = item["edge_labels"]
            K: int = item["node_labels"].shape[0]
            node_labels[b, :K] = item["node_labels"]
            node_mask[b, :K] = True

    out: dict = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "edge_mask": edge_mask,
        "struct_rel": struct_rel,
        "cols": cols_list,
    }
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels
        out["node_mask"] = node_mask
    return out


# ============================================================
# Model Architecture — v8b + Structural Attention Bias
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, d: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(n_groups, d)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.conv(x)))


class MergeOperator(nn.Module):
    def __init__(self, n_inputs: int, d: int):
        super().__init__()
        self.linear = nn.Linear(n_inputs * d, d)
        self.norm = nn.LayerNorm(d)
        self.act = nn.GELU()

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        return self.act(self.norm(self.linear(torch.cat(inputs, dim=-1))))


class StemLayer(nn.Module):
    def __init__(self, d: int, n_channels: int = None):
        super().__init__()
        n_channels = n_channels or N_CHANNELS
        self.linear = nn.Linear(n_channels, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class EdgeFeatureExtractor(nn.Module):
    def __init__(self, d: int = 64, n_blocks: int = 5, n_channels: int = None):
        super().__init__()
        self.stem = StemLayer(d, n_channels)
        self.blocks = nn.Sequential(*[ConvBlock(d) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


class StructuralSelfAttention(nn.Module):
    """Self-attention with graph-structural bias.

    Standard multi-head attention + a learned scalar bias per head
    for each of the 6 structural relationship types between edge pairs.
    The bias is added to attention logits before softmax.
    """

    def __init__(
        self, d: int = 64, n_heads: int = 4,
        n_struct_types: int = 6,
    ):
        super().__init__()
        self.d: int = d
        self.n_heads: int = n_heads
        self.head_dim: int = d // n_heads

        # Standard QKV projections
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        # Structural bias: (n_struct_types, n_heads) — one scalar per type per head
        self.struct_bias = nn.Embedding(n_struct_types, n_heads)
        nn.init.zeros_(self.struct_bias.weight)  # start neutral

        # Standard transformer block components
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.norm2 = nn.LayerNorm(d)

        self.scale: float = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,              # (B, E, d)
        struct_rel: torch.Tensor,      # (B, E, E) int64 — relationship types
        key_padding_mask: torch.Tensor | None = None,  # (B, E) bool, True=pad
    ) -> torch.Tensor:
        B, E, _ = x.shape

        # QKV
        Q: torch.Tensor = self.q_proj(x).view(B, E, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, E, hd)
        K: torch.Tensor = self.k_proj(x).view(B, E, self.n_heads, self.head_dim).transpose(1, 2)
        V: torch.Tensor = self.v_proj(x).view(B, E, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, H, E, E)
        attn_scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Structural bias: lookup (B, E, E) → (B, E, E, H) → (B, H, E, E)
        bias: torch.Tensor = self.struct_bias(struct_rel)  # (B, E, E, H)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, E, E)
        attn_scores = attn_scores + bias

        # Padding mask: set padded positions to -inf
        if key_padding_mask is not None:
            # key_padding_mask: (B, E), True where padded
            mask_expanded: torch.Tensor = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, E)
            attn_scores = attn_scores.masked_fill(mask_expanded, float('-inf'))

        attn_weights: torch.Tensor = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # handle all-masked rows

        # Apply attention
        attn_out: torch.Tensor = torch.matmul(attn_weights, V)  # (B, H, E, hd)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, E, self.d)
        attn_out = self.out_proj(attn_out)

        # Residual + FFN
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class ADIAModel(nn.Module):
    """
    v11: v8b architecture + graph-structural attention bias.
    Only change: SelfAttentionLayer → StructuralSelfAttention.
    """

    def __init__(
        self, d: int = None, n_edge_types: int = None,
        aug_noise_std: float = 0.0,
    ):
        super().__init__()
        d = d or D_MODEL
        n_edge_types = n_edge_types or N_EDGE_TYPES
        self.d: int = d
        self.aug_noise_std: float = aug_noise_std

        self.extractor = EdgeFeatureExtractor(d, n_channels=N_CHANNELS)
        self.edge_type_emb = nn.Embedding(n_edge_types, d)
        self.edge_merge = MergeOperator(n_inputs=2, d=d)

        # === THE CHANGE: structural attention instead of vanilla ===
        self.attn_layers = nn.ModuleList(
            [StructuralSelfAttention(d, n_heads=N_HEADS, n_struct_types=N_STRUCT_TYPES)
             for _ in range(2)]
        )

        self.edge_head = nn.Linear(d, 2)
        self.node_merge = MergeOperator(n_inputs=4, d=d)
        self.node_head = nn.Linear(d, N_CLASSES)

    def forward(
        self,
        edge_data: torch.Tensor,
        edge_types: torch.Tensor,
        edge_mask: torch.Tensor,
        cols_list: list[list[str]],
        struct_rel: torch.Tensor = None,
    ) -> tuple:
        B, E, C, N = edge_data.shape

        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std

        x_flat: torch.Tensor = edge_data.view(B * E, C, N)
        conv_emb: torch.Tensor = self.extractor(x_flat).view(B, E, self.d)
        type_emb: torch.Tensor = self.edge_type_emb(edge_types)
        edge_emb: torch.Tensor = self.edge_merge([conv_emb, type_emb])

        inv_mask: torch.Tensor = ~edge_mask

        # If no struct_rel provided (e.g. inference without it), build on the fly
        if struct_rel is None:
            struct_rel = torch.full(
                (B, E, E), 5, dtype=torch.long, device=edge_data.device
            )
            for b in range(B):
                p: int = len(cols_list[b])
                e: int = p * (p - 1)
                rel: np.ndarray = get_struct_rel_matrix(p)
                struct_rel[b, :e, :e] = torch.from_numpy(rel).to(edge_data.device)

        for layer in self.attn_layers:
            edge_emb = layer(edge_emb, struct_rel, key_padding_mask=inv_mask)

        edge_logits: torch.Tensor = self.edge_head(edge_emb)

        node_logits_list: list[torch.Tensor | None] = []
        for b in range(B):
            cols: list[str] = cols_list[b]
            p: int = len(cols)
            col2idx: dict[str, int] = {c: i for i, c in enumerate(cols)}
            other: list[str] = [c for c in cols if c not in ("X", "Y")]

            if not other:
                node_logits_list.append(None)
                continue

            node_logits: list[torch.Tensor] = []
            for v_name in other:
                vi: int = col2idx[v_name]
                xi: int = col2idx["X"]
                yi: int = col2idx["Y"]

                def _eidx(u: int, v: int) -> int:
                    return u * (p - 1) + v - (1 if v > u else 0)

                embs: list[torch.Tensor] = [
                    edge_emb[b, _eidx(vi, xi)],
                    edge_emb[b, _eidx(vi, yi)],
                    edge_emb[b, _eidx(xi, vi)],
                    edge_emb[b, _eidx(yi, vi)],
                ]
                node_emb: torch.Tensor = self.node_merge(embs)
                node_logits.append(self.node_head(node_emb))

            node_logits_list.append(
                torch.stack(node_logits) if node_logits else None
            )

        return edge_logits, node_logits_list


# ============================================================
# Training Wrapper
# ============================================================
def compute_class_weights(y_list: list[pd.DataFrame]) -> torch.Tensor:
    adjacency_label: dict = get_adjacency_label()
    counts: torch.Tensor = torch.zeros(N_CLASSES)
    for y_df in y_list:
        cols: list[str] = list(y_df.columns)
        arr: np.ndarray = y_df.values
        col_idx: dict[str, int] = {c: i for i, c in enumerate(cols)}
        for v in cols:
            if v in ("X", "Y"):
                continue
            idx: list[int] = [col_idx[v], col_idx["X"], col_idx["Y"]]
            sub: np.ndarray = arr[np.ix_(idx, idx)]
            key: tuple = tuple(sub.flatten())
            counts[CLASS_NAMES.index(adjacency_label.get(key, "Independent"))] += 1
    w: torch.Tensor = 1.0 / (counts + 1e-6)
    return w / w.sum() * N_CLASSES


def compute_edge_class_weights(y_list: list[pd.DataFrame]) -> torch.Tensor:
    counts: torch.Tensor = torch.zeros(2)
    for y_df in y_list:
        arr: np.ndarray = y_df.values
        p: int = arr.shape[0]
        for i in range(p):
            for j in range(p):
                if i != j:
                    counts[int(arr[i, j])] += 1
    w: torch.Tensor = 1.0 / (counts + 1e-6)
    return w / w.sum() * 2


class ADIAModelWrapper(pl.LightningModule):
    def __init__(
        self, d: int = None, node_class_weights: torch.Tensor = None,
        edge_class_weights: torch.Tensor = None,
        lr: float = 1e-3, max_epochs: int = 20,
        aug_noise_std: float = 0.0,
    ):
        super().__init__()
        d = d or D_MODEL
        self.model = ADIAModel(d, aug_noise_std=aug_noise_std)
        self.lr: float = lr
        self.max_epochs: int = max_epochs
        self.node_criterion = nn.CrossEntropyLoss(
            weight=node_class_weights if node_class_weights is not None
            else torch.ones(N_CLASSES),
            ignore_index=-1,
        )
        self.edge_criterion = nn.CrossEntropyLoss(
            weight=edge_class_weights if edge_class_weights is not None
            else torch.ones(2),
            ignore_index=-1,
        )

    def forward(self, batch: dict) -> tuple:
        return self.model(
            batch["edge_data"].to(self.device),
            batch["edge_types"].to(self.device),
            batch["edge_mask"].to(self.device),
            batch["cols"],
            struct_rel=batch["struct_rel"].to(self.device),
        )

    def _compute_loss(self, batch: dict, split: str) -> torch.Tensor:
        edge_logits, node_logits_list = self(batch)
        B: int = edge_logits.shape[0]

        total_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
        n_terms: int = 0

        if "edge_labels" in batch:
            el: torch.Tensor = batch["edge_labels"].to(self.device)
            mask: torch.Tensor = batch["edge_mask"].to(self.device)
            flat_logits: torch.Tensor = edge_logits[mask]
            flat_labels: torch.Tensor = el[mask]
            if flat_logits.numel() > 0:
                edge_loss: torch.Tensor = self.edge_criterion(flat_logits, flat_labels)
                total_loss = total_loss + edge_loss
                n_terms += 1
                self.log(f"{split}/edge_loss", edge_loss, prog_bar=False, sync_dist=True)

        if "node_labels" in batch:
            all_node_logits: list[torch.Tensor] = []
            all_node_labels: list[torch.Tensor] = []
            nl: torch.Tensor = batch["node_labels"].to(self.device)
            for b in range(B):
                if node_logits_list[b] is not None:
                    K: int = node_logits_list[b].shape[0]
                    all_node_logits.append(node_logits_list[b])
                    all_node_labels.append(nl[b, :K])
            if all_node_logits:
                cat_logits: torch.Tensor = torch.cat(all_node_logits, dim=0)
                cat_labels: torch.Tensor = torch.cat(all_node_labels, dim=0)
                valid: torch.Tensor = cat_labels >= 0
                if valid.any():
                    node_loss: torch.Tensor = self.node_criterion(
                        cat_logits[valid], cat_labels[valid]
                    )
                    total_loss = total_loss + node_loss
                    n_terms += 1
                    self.log(f"{split}/node_loss", node_loss, prog_bar=True, sync_dist=True)

        if n_terms > 0:
            total_loss = total_loss / n_terms
        self.log(f"{split}/loss", total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._compute_loss(batch, "train")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        return [optimizer], [scheduler]


# ============================================================
# Train & Infer
# ============================================================
def train(
    X_train: typing.Dict[str, pd.DataFrame],
    y_train: typing.Dict[str, pd.DataFrame],
    model_directory_path: str,
) -> None:
    keys: list[str] = list(X_train.keys())
    X_list: list[pd.DataFrame] = [X_train[k] for k in keys]
    y_list: list[pd.DataFrame] = [y_train[k] for k in keys]
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    node_w_path: str = os.path.join(LOCAL_CACHE_DIR, "node_weights_v8b.pt")
    edge_w_path: str = os.path.join(LOCAL_CACHE_DIR, "edge_weights_v8b.pt")
    if os.path.exists(node_w_path) and os.path.exists(edge_w_path):
        node_w: torch.Tensor = torch.load(node_w_path, weights_only=True)
        edge_w: torch.Tensor = torch.load(edge_w_path, weights_only=True)
    else:
        node_w = compute_class_weights(y_list)
        edge_w = compute_edge_class_weights(y_list)
        torch.save(node_w, node_w_path)
        torch.save(edge_w, edge_w_path)

    # Reuse v8b base cache
    cache_path: str = os.path.join(
        LOCAL_CACHE_DIR, f"train_dataset_v8b_anm_base_nk{N_KERNEL}.pkl"
    )
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, "rb") as f:
            samples: list[dict] = pickle.load(f)
    else:
        args: list[tuple] = [
            (X_list[i], y_list[i]) for i in range(len(X_list))
        ]
        import multiprocessing as mp
        n_workers: int = max(1, mp.cpu_count() - 1)
        print(f"Building dataset ({len(args)} samples, {n_workers} workers)...")
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            raw: list[dict] = [None] * len(args)
            futures = {
                pool.submit(_build_single, a): idx
                for idx, a in enumerate(args)
            }
            for fut in tqdm(as_completed(futures), total=len(args)):
                raw[futures[fut]] = fut.result()
        samples = raw
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    dataset = InMemoryDataset(samples)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} samples")

    wrapper = ADIAModelWrapper(
        d=D_MODEL, node_class_weights=node_w, edge_class_weights=edge_w,
        lr=LR, max_epochs=MAX_EPOCHS, aug_noise_std=AUG_NOISE_STD,
    )
    n_params: int = sum(p.numel() for p in wrapper.model.parameters())
    print(f"Params: {n_params:,}")

    trainer = pl.Trainer(
        accelerator="gpu", devices=2, strategy="ddp",
        max_epochs=MAX_EPOCHS, precision="32-true",
        use_distributed_sampler=True,
        logger=True, enable_checkpointing=True, enable_progress_bar=True,
    )
    trainer.fit(wrapper, loader)

    path: str = os.path.join(model_directory_path, "model.pt")
    sd: dict = wrapper.model.state_dict()
    torch.save(
        {k.replace("module.", ""): v for k, v in sd.items()}, path
    )
    print(f"Model saved to {path}")


@torch.no_grad()
def infer_batch_local(dfs, model, device="cpu", batch_size=32, cache_dir=None):
    model = model.eval()
    cache_path = os.path.join(cache_dir, f"infer_v8b_nk{N_KERNEL}.pkl") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            all_items = pickle.load(f)
    else:
        import multiprocessing as mp
        n_workers = max(1, mp.cpu_count() - 1)
        args = [(df, None) for df in dfs]
        print(f"Building edge tensors ({len(dfs)} graphs, {n_workers} workers)...")
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            all_items = list(tqdm(pool.map(_build_single, args, chunksize=8), total=len(args)))
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
        edge_logits, node_logits_list = model(
            batch["edge_data"].to(device), batch["edge_types"].to(device),
            batch["edge_mask"].to(device), batch["cols"],
            struct_rel=batch["struct_rel"].to(device))
        for b in range(edge_logits.shape[0]):
            item = all_items[idx_off + b]
            cols, p = item["cols"], item["p"]
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
                for k, nn_ in enumerate(other_nodes):
                    A.loc[nn_, :] = 0
                    A.loc[:, nn_] = 0
                    for (s, d) in patterns[CLASS_NAMES[node_preds[k].item()]](nn_):
                        A.loc[s, d] = 1
            results[idx_off + b] = A
        idx_off += edge_logits.shape[0]
    return results


def infer(X_test, model_directory_path, id_column_name, prediction_column_name):
    path = os.path.join(model_directory_path, "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL, aug_noise_std=0.0)
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
# K-Fold Train + Eval (works on preprocessed samples)
# ============================================================
def train_on_samples(
    samples: list[dict], node_w: torch.Tensor, edge_w: torch.Tensor,
    save_path: str, seed: int = 42,
) -> None:
    """Train on a list of preprocessed samples."""
    pl.seed_everything(seed)
    dataset = InMemoryDataset(samples)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )
    wrapper = ADIAModelWrapper(
        d=D_MODEL, node_class_weights=node_w, edge_class_weights=edge_w,
        lr=LR, max_epochs=MAX_EPOCHS, aug_noise_std=AUG_NOISE_STD,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=2, strategy="ddp",
        max_epochs=MAX_EPOCHS, precision="32-true",
        use_distributed_sampler=True,
        logger=False, enable_checkpointing=False, enable_progress_bar=True,
    )
    trainer.fit(wrapper, loader)
    sd = wrapper.model.state_dict()
    torch.save(
        {k.replace("module.", ""): v for k, v in sd.items()}, save_path
    )


def eval_on_samples(
    val_samples: list[dict], model_path: str, device: str = "cuda",
) -> tuple[float, dict[str, float]]:
    """Evaluate on preprocessed samples. Returns (balanced_acc, per_class_dict)."""
    model = ADIAModel(d=D_MODEL, aug_noise_std=0.0)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    loader = DataLoader(
        InMemoryDataset(val_samples), batch_size=64,
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

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
    adjacency_label = get_adjacency_label()

    cc: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    ct: dict[str, int] = {c: 0 for c in CLASS_NAMES}

    idx_off: int = 0
    with torch.no_grad():
        for batch in loader:
            edge_logits, node_logits_list = model(
                batch["edge_data"].to(device), batch["edge_types"].to(device),
                batch["edge_mask"].to(device), batch["cols"],
                struct_rel=batch["struct_rel"].to(device),
            )
            B = len(batch["cols"])
            for b in range(B):
                item = val_samples[idx_off + b]
                cols = item["cols"]
                p = item["p"]

                # Get true labels
                if "node_labels" not in item:
                    idx_off += B
                    continue
                other_nodes = item.get("other_nodes", [c for c in cols if c not in ("X", "Y")])
                true_labels = item["node_labels"]

                # Get predictions
                if node_logits_list[b] is not None:
                    node_preds = torch.argmax(node_logits_list[b], dim=-1).cpu().numpy()
                else:
                    node_preds = np.full(len(other_nodes), CLASS_NAMES.index("Independent"))

                for k, v_name in enumerate(other_nodes):
                    true_cls = CLASS_NAMES[true_labels[k]]
                    pred_cls = CLASS_NAMES[node_preds[k]]
                    ct[true_cls] += 1
                    if true_cls == pred_cls:
                        cc[true_cls] += 1

            idx_off += B

    per_class: dict[str, float] = {}
    accs: list[float] = []
    for cls in CLASS_NAMES:
        n = ct[cls]
        acc = cc[cls] / n if n > 0 else 0.0
        per_class[cls] = acc
        accs.append(acc)

    return float(np.mean(accs)), per_class


# ============================================================
# Main — 5-Fold CV
# ============================================================
N_FOLDS: int = 5

if __name__ == "__main__":
    print("=" * 60)
    print("v11 Structural Attention Bias — 5-Fold Cross-Validation")
    print(f"  6 struct types, {N_HEADS} heads, d={D_MODEL}")
    print(f"  Config: bs={BATCH_SIZE}, lr={LR}, epochs={MAX_EPOCHS}")
    print(f"  Folds: {N_FOLDS}")
    print("=" * 60)

    X_train = pd.read_pickle("data/X_train.pickle")
    y_train = pd.read_pickle("data/y_train.pickle")
    print(f"Loaded {len(X_train)} training samples.")

    # Class weights (from full dataset)
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    node_w_path = os.path.join(LOCAL_CACHE_DIR, "node_weights_v8b.pt")
    edge_w_path = os.path.join(LOCAL_CACHE_DIR, "edge_weights_v8b.pt")
    if os.path.exists(node_w_path) and os.path.exists(edge_w_path):
        node_w = torch.load(node_w_path, weights_only=True)
        edge_w = torch.load(edge_w_path, weights_only=True)
    else:
        y_list = [y_train[k] for k in y_train]
        node_w = compute_class_weights(y_list)
        edge_w = compute_edge_class_weights(y_list)
        torch.save(node_w, node_w_path)
        torch.save(edge_w, edge_w_path)

    # Load or build full preprocessed dataset
    cache_path = os.path.join(
        LOCAL_CACHE_DIR, f"train_dataset_v8b_anm_base_nk{N_KERNEL}.pkl"
    )
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, "rb") as f:
            all_samples = pickle.load(f)
    else:
        keys = list(X_train.keys())
        args = [(X_train[k], y_train[k]) for k in keys]
        import multiprocessing as mp
        n_workers = max(1, mp.cpu_count() - 1)
        print(f"Building dataset ({len(args)} samples, {n_workers} workers)...")
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            all_samples = [None] * len(args)
            futures = {pool.submit(_build_single, a): idx for idx, a in enumerate(args)}
            for fut in tqdm(as_completed(futures), total=len(args)):
                all_samples[futures[fut]] = fut.result()
        with open(cache_path, "wb") as f:
            pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Total samples: {len(all_samples)}")

    # Shuffle and split into folds
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    fold_size = len(all_samples) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        start = i * fold_size
        end = start + fold_size if i < N_FOLDS - 1 else len(all_samples)
        folds.append(indices[start:end])

    # Run k-fold — train all folds first (both DDP ranks)
    for fold_i in range(N_FOLDS):
        print(f"\n{'='*50} Training Fold {fold_i+1}/{N_FOLDS} {'='*50}")

        val_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_i])

        train_samples = [all_samples[i] for i in train_idx]
        print(f"  Train: {len(train_samples)}, Val: {len(val_idx)}")

        model_path = os.path.join("resources", f"model_fold{fold_i}.pt")
        train_on_samples(train_samples, node_w, edge_w, model_path, seed=42 + fold_i)

    # DDP guard — only rank 0 does evaluation
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank != 0:
        exit(0)

    # Evaluate all folds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_accs: list[float] = []
    fold_per_class: list[dict] = []

    for fold_i in range(N_FOLDS):
        print(f"\n{'='*50} Evaluating Fold {fold_i+1}/{N_FOLDS} {'='*50}")
        val_idx = folds[fold_i]
        val_samples = [all_samples[i] for i in val_idx]
        model_path = os.path.join("resources", f"model_fold{fold_i}.pt")

        bal_acc, per_class = eval_on_samples(val_samples, model_path, device)
        fold_accs.append(bal_acc)
        fold_per_class.append(per_class)

        print(f"  Fold {fold_i+1} Balanced Accuracy: {bal_acc:.4f}")
        for cls in CLASS_NAMES:
            print(f"    {cls:25s}: {per_class[cls]:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("5-FOLD CV SUMMARY")
    print("=" * 60)
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i+1}: {acc:.4f}")
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n  Mean: {mean_acc:.4f} ± {std_acc:.4f}")

    # Per-class mean ± std
    print("\nPer-class mean ± std:")
    for cls in CLASS_NAMES:
        cls_accs = [fp[cls] for fp in fold_per_class]
        print(f"  {cls:25s}: {np.mean(cls_accs):.4f} ± {np.std(cls_accs):.4f}")
