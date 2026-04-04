"""
v18_binary_decomp.py — ADIA Causal Discovery
  v11 architecture + edge-based binary decomposition at inference

Core insight: The 8 node classes map perfectly to a 4-bit binary code
representing presence/absence of edges [v->X, v->Y, X->v, Y->v].
The edge_head already predicts each edge's probability. By computing
class scores from these 4 edge probabilities (Naive Bayes decomposition),
we get a second opinion that directly captures conjunctive patterns
like Mediator (requires BOTH X->v AND v->Y).

Combining node_head logits with binary decomposition scores:
  final = node_logits + alpha * binary_scores

Zero retraining — uses v11 checkpoint directly. Sweeps alpha.

Usage:
    python v18_binary_decomp.py
"""

# @crunch/keep:on
import crunch

import os

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
N_STRUCT_TYPES: int = 6
CLASS_NAMES: list[str] = [
    "Confounder", "Collider", "Mediator",
    "Cause of X", "Cause of Y",
    "Consequence of X", "Consequence of Y",
    "Independent",
]

# v18: Binary edge pattern for each class
# Columns: [v->X, v->Y, X->v, Y->v]
CLASS_EDGE_PATTERN: torch.Tensor = torch.tensor([
    [1, 1, 0, 0],  # 0: Confounder   — v causes X and Y
    [0, 0, 1, 1],  # 1: Collider     — X and Y cause v
    [0, 1, 1, 0],  # 2: Mediator     — X->v, v->Y
    [1, 0, 0, 0],  # 3: Cause of X   — v->X only
    [0, 1, 0, 0],  # 4: Cause of Y   — v->Y only
    [0, 0, 1, 0],  # 5: Consequence of X — X->v only
    [0, 0, 0, 1],  # 6: Consequence of Y — Y->v only
    [0, 0, 0, 0],  # 7: Independent  — no edges
], dtype=torch.float32)  # (8, 4)

BANDWIDTHS: list[float] = [0.2, 0.5, 1.0]
N_CHANNELS: int = 2 + len(BANDWIDTHS) + len(BANDWIDTHS)  # 8

MAX_EPOCHS: int = 30
BATCH_SIZE: int = 16
LR: float = 1e-3
AUG_NOISE_STD: float = 0.01
LOCAL_CACHE_DIR: str = "dataset_cache/"
IS_CLOUD_SUBMIT: bool = False


# ============================================================
# Graph Utilities (identical to v11)
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
# Data Preprocessing (identical to v11)
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
# Structural Relationship Matrix (identical to v11)
# ============================================================
def build_struct_rel_matrix(p: int) -> np.ndarray:
    E: int = p * (p - 1)
    edge_uv: list[tuple[int, int]] = []
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
# Dataset & Collate (identical to v11)
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
# Model Architecture (identical to v11)
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
    def __init__(self, d: int = 64, n_heads: int = 4, n_struct_types: int = 6):
        super().__init__()
        self.d: int = d
        self.n_heads: int = n_heads
        self.head_dim: int = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        self.struct_bias = nn.Embedding(n_struct_types, n_heads)
        nn.init.zeros_(self.struct_bias.weight)

        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.norm2 = nn.LayerNorm(d)
        self.scale: float = self.head_dim ** -0.5

    def forward(self, x, struct_rel, key_padding_mask=None):
        B, E, _ = x.shape
        Q = self.q_proj(x).view(B, E, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, E, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, E, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        bias = self.struct_bias(struct_rel).permute(0, 3, 1, 2)
        attn_scores = attn_scores + bias

        if key_padding_mask is not None:
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded, float('-inf'))

        attn_weights = torch.nan_to_num(F.softmax(attn_scores, dim=-1), nan=0.0)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, E, self.d)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class ADIAModel(nn.Module):
    """v11 architecture — identical, loaded from v11 checkpoint."""

    def __init__(self, d: int = None, n_edge_types: int = None, aug_noise_std: float = 0.0):
        super().__init__()
        d = d or D_MODEL
        n_edge_types = n_edge_types or N_EDGE_TYPES
        self.d: int = d
        self.aug_noise_std: float = aug_noise_std

        self.extractor = EdgeFeatureExtractor(d, n_channels=N_CHANNELS)
        self.edge_type_emb = nn.Embedding(n_edge_types, d)
        self.edge_merge = MergeOperator(n_inputs=2, d=d)

        self.attn_layers = nn.ModuleList(
            [StructuralSelfAttention(d, n_heads=N_HEADS, n_struct_types=N_STRUCT_TYPES)
             for _ in range(2)]
        )

        self.edge_head = nn.Linear(d, 2)
        self.node_merge = MergeOperator(n_inputs=4, d=d)
        self.node_head = nn.Linear(d, N_CLASSES)

    def forward_full(
        self,
        edge_data: torch.Tensor,
        edge_types: torch.Tensor,
        edge_mask: torch.Tensor,
        cols_list: list[list[str]],
        struct_rel: torch.Tensor = None,
        alpha: float = 0.0,
    ) -> tuple:
        """v18: Returns edge_logits and node predictions.

        When alpha > 0, combines node_head logits with binary decomposition
        scores from the edge_head.
        """
        B, E, C, N = edge_data.shape

        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std

        x_flat = edge_data.view(B * E, C, N)
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

        edge_logits = self.edge_head(edge_emb)

        # Precompute pattern tensor on correct device
        pattern = CLASS_EDGE_PATTERN.to(edge_data.device)  # (8, 4)

        node_logits_list: list[torch.Tensor | None] = []
        for b in range(B):
            cols = cols_list[b]
            p = len(cols)
            col2idx = {c: i for i, c in enumerate(cols)}
            other = [c for c in cols if c not in ("X", "Y")]

            if not other:
                node_logits_list.append(None)
                continue

            node_logits_batch: list[torch.Tensor] = []
            for v_name in other:
                vi = col2idx[v_name]
                xi = col2idx["X"]
                yi = col2idx["Y"]

                def _eidx(u: int, v: int) -> int:
                    return u * (p - 1) + v - (1 if v > u else 0)

                emb_vx = edge_emb[b, _eidx(vi, xi)]
                emb_vy = edge_emb[b, _eidx(vi, yi)]
                emb_xv = edge_emb[b, _eidx(xi, vi)]
                emb_yv = edge_emb[b, _eidx(yi, vi)]

                # Standard node head (v11)
                node_emb = self.node_merge([emb_vx, emb_vy, emb_xv, emb_yv])
                logits = self.node_head(node_emb)

                if alpha > 0:
                    # v18: Binary decomposition from edge_head
                    edge_probs_4 = torch.stack([
                        F.softmax(edge_logits[b, _eidx(vi, xi)], dim=-1)[1],  # P(v->X)
                        F.softmax(edge_logits[b, _eidx(vi, yi)], dim=-1)[1],  # P(v->Y)
                        F.softmax(edge_logits[b, _eidx(xi, vi)], dim=-1)[1],  # P(X->v)
                        F.softmax(edge_logits[b, _eidx(yi, vi)], dim=-1)[1],  # P(Y->v)
                    ])  # (4,)

                    # Clamp to avoid log(0)
                    p_edge = edge_probs_4.clamp(min=1e-6, max=1 - 1e-6)
                    log_p = torch.log(p_edge)        # (4,)
                    log_1mp = torch.log(1 - p_edge)  # (4,)

                    # Binary score for each class: sum(pattern * log(p) + (1-pattern) * log(1-p))
                    binary_scores = (pattern * log_p + (1 - pattern) * log_1mp).sum(dim=-1)  # (8,)

                    logits = logits + alpha * binary_scores

                node_logits_batch.append(logits)

            node_logits_list.append(
                torch.stack(node_logits_batch) if node_logits_batch else None
            )

        return edge_logits, node_logits_list

    def forward(self, edge_data, edge_types, edge_mask, cols_list, struct_rel=None):
        """Standard forward (v11 compatible, alpha=0)."""
        return self.forward_full(edge_data, edge_types, edge_mask, cols_list, struct_rel, alpha=0.0)


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def infer_batch_local(dfs, model, device="cpu", batch_size=32, cache_dir=None, alpha=0.0):
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
    for batch in tqdm(loader, desc=f"inferring (alpha={alpha})"):
        edge_logits, node_logits_list = model.forward_full(
            batch["edge_data"].to(device), batch["edge_types"].to(device),
            batch["edge_mask"].to(device), batch["cols"],
            struct_rel=batch["struct_rel"].to(device),
            alpha=alpha)
        B_cur = edge_logits.shape[0]
        for b in range(B_cur):
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
        idx_off += B_cur
    return results


def evaluate(adj_list, names, y_test):
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
    accs = []
    for cls in CLASS_NAMES:
        n = ct[cls]
        acc = cc[cls] / n if n > 0 else 0.0
        accs.append(acc)
    return np.mean(accs), {cls: (cc[cls] / ct[cls] if ct[cls] > 0 else 0.0, ct[cls]) for cls in CLASS_NAMES}


def infer(X_test, model_directory_path, id_column_name, prediction_column_name):
    path = os.path.join(model_directory_path, "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL, aug_noise_std=0.0)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    dfs = [X_test[n] for n in X_test]
    cache_dir = None if IS_CLOUD_SUBMIT else LOCAL_CACHE_DIR
    adj_list = infer_batch_local(dfs, model, device=device, batch_size=64,
                                  cache_dir=cache_dir, alpha=1.0)
    submission = {}
    for name, A in zip(X_test.keys(), adj_list):
        for i in A.columns:
            for j in A.columns:
                submission[f"{name}_{i}_{j}"] = int(A.loc[i, j])
    s = pd.Series(submission).reset_index()
    s.columns = [id_column_name, prediction_column_name]
    return s


# ============================================================
# Main — Alpha sweep
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("v18: Binary Edge Decomposition (inference-only)")
    print("  Combines node_head + edge-based class scoring")
    print("  final_logits = node_logits + alpha * binary_scores")
    print("  Uses v11 checkpoint — NO retraining!")
    print("=" * 60)

    X_test = pd.read_pickle("data/X_test_reduced.pickle")
    y_test = pd.read_pickle("data/y_test_reduced.pickle")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL, aug_noise_std=0.0)
    model.load_state_dict(torch.load("resources/model_v11_structbias.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    dfs = [X_test[n] for n in X_test]
    names = list(X_test.keys())

    # Sweep alpha values
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

    print(f"\nSweeping alpha over {alphas}...")
    print(f"{'alpha':>6s} | {'BalAcc':>7s} | {'Conf':>6s} {'Coll':>6s} {'Med':>6s} {'CauX':>6s} {'CauY':>6s} {'CnqX':>6s} {'CnqY':>6s} {'Indep':>6s}")
    print("-" * 90)

    best_alpha = 0.0
    best_ba = 0.0

    for alpha in alphas:
        adj_list = infer_batch_local(dfs, model, device=device, batch_size=64,
                                      cache_dir=LOCAL_CACHE_DIR, alpha=alpha)
        ba, per_class = evaluate(adj_list, names, y_test)

        accs = [per_class[c][0] for c in CLASS_NAMES]
        print(f"{alpha:6.2f} | {ba:7.4f} | {' '.join(f'{a:.4f}' for a in accs)}")

        if ba > best_ba:
            best_ba = ba
            best_alpha = alpha

    print(f"\nBest: alpha={best_alpha}, balanced accuracy={best_ba:.4f}")
    print(f"v11 baseline (alpha=0): compare row above")