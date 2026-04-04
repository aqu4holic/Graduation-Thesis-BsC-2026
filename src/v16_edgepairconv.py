"""
v16_edgepairconv.py — Architecture A: Edge-Pair Convolution

Novel architecture: adds a parallel NodeConvBranch that processes the 4 raw
edge curves per node (v→X, v→Y, X→v, Y→v) jointly via 1D convolution.

Key insight: v11's node head receives 4 edge embeddings AFTER conv1d has
compressed each edge independently to 64d. By the time the model can reason
"X→v is active AND v→Y is active → Mediator", the raw functional evidence
is gone. NodeConv learns cross-edge patterns directly from the raw curves.

Architecture:
  Path 1 (v11 backbone, unchanged):
    8ch edge tensors → Stem → 5×Conv1d → AvgPool → Merge[conv, type_emb]
    → 2× StructuralSelfAttention → edge_emb (B, E, 64)

  Path 2 (NEW — NodeConvBranch):
    For each node v, gather 4 raw edge tensors (v→X, v→Y, X→v, Y→v)
    → stack to (4×8, N) = (32, N) → Stem(32→64) → 3×ConvBlock → AvgPool
    → 64-dim node embedding from raw cross-edge patterns

  Fusion Node Head:
    v11_merge(4 gathered edge_embs) → 64d
    nodeconv_emb → 64d
    v11_node + nodeconv_gate(nodeconv) → classifier → 8 classes

  Edge head: unchanged from v11 (Tower 1 only)

Preprocessing: identical to v11 (8ch), reuses same cache.
Cache tag: v8b_anm_base (same as v11).

Thesis contribution: "Direct pairwise edge convolution for causal role
classification — the first architecture to learn cross-edge functional
patterns from raw sorted observation curves."

Usage:
    python v16_edgepairconv.py
"""

# @crunch/keep:on
import crunch

import os

import typing
import glob
import gc
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.nn.functional as F

import pytorch_lightning as pl
import torch.autograd.graph
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

import networkx as nx
from sklearn.model_selection import train_test_split

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

# === Multi-bandwidth config ===
BANDWIDTHS: list[float] = [0.2, 0.5, 1.0]

# Total channels: 2 (sorted obs) + 3 (kernel bw) + 3 (ANM resid bw) = 8
N_CHANNELS: int = 2 + len(BANDWIDTHS) + len(BANDWIDTHS)

# === NodeConv config ===
N_NODECONV_CHANNELS: int = 4 * N_CHANNELS  # 4 edges × 8 channels = 32
N_NODECONV_BLOCKS: int = 3  # lighter than the 5-block edge extractor

# === Training config ===
MAX_EPOCHS: int = 30
BATCH_SIZE: int = 64
LR: float = 2e-3
AUG_NOISE_STD: float = 0.01
N_AUG: int = 1
LOSS_EDGE_W: float = 0.3

IS_CLOUD_SUBMIT: bool = False
LOCAL_CACHE_DIR: str = "dataset_cache"

# ============================================================
# Label / Graph Utilities
# ============================================================
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


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


# ============================================================
# Kernel Regression + ANM (identical to v11/v8b)
# ============================================================
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


# ============================================================
# Edge Tensor Builder (identical to v11)
# ============================================================
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
            edge_tensor: np.ndarray = np.stack(channels, axis=0)  # (8, N)
            edges.append(edge_tensor)
            edge_types.append(_edge_type(u_name, v_name))

    edge_data: np.ndarray = np.stack(edges, axis=0).astype(np.float32)
    edge_types_arr: np.ndarray = np.array(edge_types, dtype=np.int64)
    return edge_data, edge_types_arr


# ============================================================
# Structural Relationship Matrix (v11)
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
# Sample Builder + Dataset (identical to v11)
# ============================================================
def _build_single(args: tuple) -> dict:
    df, y_df = args
    edge_data, edge_types = build_edge_tensor(df)
    cols: list[str] = list(df.columns)
    result: dict = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "cols": cols,
    }
    if y_df is not None:
        adj: np.ndarray = y_df.values

        p: int = len(cols)
        edge_labels: list[int] = []
        for i in range(p):
            for j in range(p):
                if i != j:
                    edge_labels.append(int(adj[i, j] != 0))
        result["edge_labels"] = np.array(edge_labels, dtype=np.int64)

        adjacency_label: dict = get_adjacency_label()
        labels: dict = get_labels(y_df, adjacency_label)
        node_labels: list[int] = []
        for c in cols:
            if c in ("X", "Y"):
                continue
            label_name: str = labels.get(c, "Independent")
            node_labels.append(CLASS_TO_IDX[label_name])
        result["node_labels"] = np.array(node_labels, dtype=np.int64)

    return result


class InMemoryDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples: list[dict] = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    B: int = len(batch)
    max_E: int = max(item["edge_data"].shape[0] for item in batch)
    C: int = batch[0]["edge_data"].shape[1]
    N: int = batch[0]["edge_data"].shape[2]
    max_K: int = max(
        (item["node_labels"].shape[0] if "node_labels" in item else 0)
        for item in batch
    )
    if max_K == 0:
        max_K = 1

    edge_data: torch.Tensor = torch.zeros(B, max_E, C, N)
    edge_types: torch.Tensor = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask: torch.Tensor = torch.zeros(B, max_E, dtype=torch.bool)
    edge_labels: torch.Tensor = torch.full((B, max_E), -1, dtype=torch.long)
    node_labels: torch.Tensor = torch.full((B, max_K), -1, dtype=torch.long)
    node_mask: torch.Tensor = torch.zeros(B, max_K, dtype=torch.bool)

    cols_list: list = []
    has_labels: bool = False

    # Build struct_rel matrices
    struct_rel_list: list[np.ndarray] = []
    max_struct_E: int = 0
    for item in batch:
        p: int = len(item["cols"])
        sr: np.ndarray = get_struct_rel_matrix(p)
        struct_rel_list.append(sr)
        max_struct_E = max(max_struct_E, sr.shape[0])

    struct_rel: torch.Tensor = torch.full(
        (B, max_struct_E, max_struct_E), N_STRUCT_TYPES - 1, dtype=torch.long
    )

    for b, item in enumerate(batch):
        E: int = item["edge_data"].shape[0]
        edge_data[b, :E] = torch.from_numpy(item["edge_data"])
        edge_types[b, :E] = torch.from_numpy(item["edge_types"])
        edge_mask[b, :E] = True
        cols_list.append(item["cols"])

        sr = struct_rel_list[b]
        sE: int = sr.shape[0]
        struct_rel[b, :sE, :sE] = torch.from_numpy(sr)

        if "edge_labels" in item:
            has_labels = True
            edge_labels[b, :E] = torch.from_numpy(item["edge_labels"])
            K: int = item["node_labels"].shape[0]
            node_labels[b, :K] = torch.from_numpy(item["node_labels"])
            node_mask[b, :K] = True

    out: dict = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "edge_mask": edge_mask,
        "cols": cols_list,
        "struct_rel": struct_rel,
    }
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels
        out["node_mask"] = node_mask
    return out


# ============================================================
# Model Architecture — v11 Backbone + NodeConv Branch
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
    """Self-attention with graph-structural bias (v11)."""
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
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.scale: float = self.head_dim ** -0.5

    def forward(
        self, x: torch.Tensor, struct_rel: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, E, self.d)
        attn_out = self.out_proj(attn_out)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ============================================================
# NEW: NodeConvBranch — Cross-Edge Pattern Detector
# ============================================================
class NodeConvBranch(nn.Module):
    """
    Processes the 4 raw edge curves per node jointly via 1D convolution.

    For each node v, takes the 4 edges (v→X, v→Y, X→v, Y→v), each (C, N).
    Stacks to (4*C, N) = (32, N), then:
      Stem(32 → d) → 3×ConvBlock → AvgPool → d-dim embedding

    The key insight: the Stem's learned projection can combine channels
    across edges (e.g., learn that channel 3 of edge v→X minus channel 3
    of edge X→v indicates directional asymmetry), and the ConvBlocks
    learn spatial patterns in these cross-edge combinations.

    This is lighter than the full 5-block EdgeFeatureExtractor (3 blocks)
    because cross-edge patterns are simpler than per-edge functional shapes.
    """

    def __init__(self, d: int = 64, n_input_channels: int = None,
                 n_blocks: int = None):
        super().__init__()
        n_input_channels = n_input_channels or N_NODECONV_CHANNELS
        n_blocks = n_blocks or N_NODECONV_BLOCKS
        self.stem = nn.Linear(n_input_channels, d)
        self.blocks = nn.Sequential(*[ConvBlock(d) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (K, 4*C, N) — K nodes, each with 4 stacked edge curves
        Returns:
            (K, d) — node embeddings from cross-edge patterns
        """
        # Stem: (K, 4*C, N) → permute → (K, N, 4*C) → linear → (K, N, d) → permute
        x = self.stem(x.permute(0, 2, 1)).permute(0, 2, 1)  # (K, d, N)
        x = self.blocks(x)  # (K, d, N)
        return self.pool(x).squeeze(-1)  # (K, d)


# ============================================================
# Full Model — v16: v11 + NodeConv Branch
# ============================================================
class ADIAModel(nn.Module):
    """
    v16: v11 backbone + parallel NodeConvBranch with zero-init residual.

    Pipeline:
      1. EdgeFeatureExtractor: (B,E,8,N) → (B,E,d)
      2. Merge [conv, type_emb] → (B,E,d)
      3. 2× StructuralSelfAttention → edge_emb (B,E,d)
      4. Edge head: edge_emb → (B,E,2)
      5. For each node v:
         a. v11 path: gather 4 edge_embs → node_merge → 64d
         b. NodeConv path: gather 4 raw edge curves → NodeConv → 64d
         c. Zero-init gate: nodeconv_gate(nodeconv_emb) starts at 0
         d. Residual fusion: v11_node + gated_nodeconv → classifier → 8

    At initialization: gated_nodeconv = 0, so v16 = v11 exactly.
    During training: gate learns to incorporate NodeConv if beneficial.
    """

    def __init__(self, d: int = None, n_edge_types: int = None,
                 aug_noise_std: float = AUG_NOISE_STD):
        super().__init__()
        d = d or D_MODEL
        n_edge_types = n_edge_types or N_EDGE_TYPES
        self.d: int = d
        self.aug_noise_std: float = aug_noise_std

        # --- Path 1: v11 edge backbone (unchanged) ---
        self.extractor = EdgeFeatureExtractor(d, n_blocks=5)
        self.edge_type_emb = nn.Embedding(n_edge_types, d)
        self.edge_merge = MergeOperator(n_inputs=2, d=d)
        self.attn_layers = nn.ModuleList([
            StructuralSelfAttention(d, n_heads=N_HEADS) for _ in range(2)
        ])
        self.edge_head = nn.Linear(d, 2)

        # --- Path 2: NodeConv branch (NEW) ---
        self.node_conv = NodeConvBranch(d)

        # --- Node head with zero-init residual ---
        self.node_merge = MergeOperator(n_inputs=4, d=d)     # v11's 4-edge merge
        self.nodeconv_gate = nn.Linear(d, d, bias=True)       # projects nodeconv → d
        nn.init.zeros_(self.nodeconv_gate.weight)              # zero-init: starts at 0
        nn.init.zeros_(self.nodeconv_gate.bias)
        self.node_classifier = nn.Linear(d, N_CLASSES)        # v11's classifier

    def forward(
        self,
        edge_data: torch.Tensor,    # (B, E, C, N)
        edge_types: torch.Tensor,   # (B, E)
        edge_mask: torch.Tensor,    # (B, E) bool
        cols_list: list,
        struct_rel: torch.Tensor = None,  # (B, E, E) int64
    ) -> tuple[torch.Tensor, list]:
        B, E, C, N = edge_data.shape

        # Apply noise augmentation (training only)
        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std

        # === Path 1: v11 edge backbone ===
        edge_emb: torch.Tensor = self.extractor(
            edge_data.view(B * E, C, N)
        ).view(B, E, self.d)
        type_emb: torch.Tensor = self.edge_type_emb(edge_types)
        edge_emb = self.edge_merge([edge_emb, type_emb])

        pad_mask: torch.Tensor = ~edge_mask
        for attn in self.attn_layers:
            edge_emb = attn(edge_emb, struct_rel=struct_rel, key_padding_mask=pad_mask)

        # Edge head (v11 only)
        edge_logits: torch.Tensor = self.edge_head(edge_emb)

        # === Fused Node Head ===
        node_logits_list: list = []
        for b in range(B):
            cols: list[str] = cols_list[b]
            p: int = len(cols)
            col_idx: dict = {name: i for i, name in enumerate(cols)}

            # Build edge ordering map
            edge_order: dict = {}
            count: int = 0
            for ui in range(p):
                for vi in range(p):
                    if ui != vi:
                        edge_order[(ui, vi)] = count
                        count += 1

            x_idx: int | None = col_idx.get("X")
            y_idx: int | None = col_idx.get("Y")
            other_nodes: list[str] = [n for n in cols if n not in ("X", "Y")]

            if not other_nodes or x_idx is None or y_idx is None:
                node_logits_list.append(None)
                continue

            # Gather raw edge data and edge embeddings for all nodes at once
            raw_stacks: list[torch.Tensor] = []
            v11_gathered: list[torch.Tensor] = []

            for node_name in other_nodes:
                u: int = col_idx[node_name]
                idx_vx: int = edge_order[(u, x_idx)]
                idx_vy: int = edge_order[(u, y_idx)]
                idx_xv: int = edge_order[(x_idx, u)]
                idx_yv: int = edge_order[(y_idx, u)]

                # Path 1: gather 4 edge embeddings → v11 merge
                embs = edge_emb[b]
                e_vx = embs[idx_vx]
                e_vy = embs[idx_vy]
                e_xv = embs[idx_xv]
                e_yv = embs[idx_yv]
                v11_node = self.node_merge([e_vx, e_vy, e_xv, e_yv])
                v11_gathered.append(v11_node)

                # Path 2: gather 4 raw edge tensors → stack to (32, N)
                raw_vx = edge_data[b, idx_vx]  # (C, N)
                raw_vy = edge_data[b, idx_vy]
                raw_xv = edge_data[b, idx_xv]
                raw_yv = edge_data[b, idx_yv]
                raw_stack = torch.cat([raw_vx, raw_vy, raw_xv, raw_yv], dim=0)  # (4*C, N)
                raw_stacks.append(raw_stack)

            # Batch NodeConv for all nodes in this graph
            raw_batch: torch.Tensor = torch.stack(raw_stacks)  # (K, 32, N)
            nodeconv_embs: torch.Tensor = self.node_conv(raw_batch)  # (K, d)

            # Zero-init residual fusion: v11_node + gate(nodeconv)
            node_logits: list[torch.Tensor] = []
            for k in range(len(other_nodes)):
                gated_nodeconv = self.nodeconv_gate(nodeconv_embs[k])  # starts at 0
                fused = v11_gathered[k] + gated_nodeconv  # starts as pure v11
                logits = self.node_classifier(fused)
                node_logits.append(logits)

            node_logits_list.append(
                torch.stack(node_logits) if node_logits else None
            )

        return edge_logits, node_logits_list


# ============================================================
# Lightning Module
# ============================================================
def compute_class_weights(y_list: list) -> torch.Tensor:
    counts: torch.Tensor = torch.zeros(N_CLASSES)
    adjacency_label: dict = get_adjacency_label()
    for y_df in y_list:
        labels: dict = get_labels(y_df, adjacency_label)
        for v, lbl in labels.items():
            counts[CLASS_TO_IDX[lbl]] += 1
    w: torch.Tensor = 1.0 / (counts + 1e-6)
    return w / w.sum() * N_CLASSES


def compute_edge_weights(y_list: list) -> torch.Tensor:
    counts: torch.Tensor = torch.zeros(2)
    for y_df in y_list:
        arr: np.ndarray = y_df.values
        p: int = arr.shape[0]
        for i in range(p):
            for j in range(p):
                if i != j:
                    counts[int(arr[i, j] != 0)] += 1
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
                total_loss = total_loss + LOSS_EDGE_W * edge_loss
                n_terms += 1
                self.log(f"{split}_edge_loss", edge_loss, prog_bar=False, sync_dist=True)

        if "node_labels" in batch:
            all_nl: list[torch.Tensor] = []
            all_tl: list[torch.Tensor] = []
            nl_batch: torch.Tensor = batch["node_labels"].to(self.device)
            for b in range(B):
                nl = node_logits_list[b]
                if nl is None:
                    continue
                K: int = nl.shape[0]
                tl: torch.Tensor = nl_batch[b, :K]
                valid: torch.Tensor = tl >= 0
                if valid.any():
                    all_nl.append(nl[valid])
                    all_tl.append(tl[valid])
            if all_nl:
                all_nl_cat: torch.Tensor = torch.cat(all_nl)
                all_tl_cat: torch.Tensor = torch.cat(all_tl)
                node_loss: torch.Tensor = self.node_criterion(all_nl_cat, all_tl_cat)
                total_loss = total_loss + node_loss
                n_terms += 1
                self.log(f"{split}_node_loss", node_loss, prog_bar=True, sync_dist=True)

                # Balanced accuracy tracking
                preds = all_nl_cat.argmax(dim=-1)
                n_unique = preds.unique().numel()
                self.log(f"{split}_n_unique_preds", float(n_unique), prog_bar=True, sync_dist=True)

        if n_terms == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log(f"{split}_loss", total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._compute_loss(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        return [optimizer], [scheduler]


# ============================================================
# Training
# ============================================================
def train(X_train, y_train, model_directory_path: str = "resources"):
    os.makedirs(model_directory_path, exist_ok=True)
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    X_list: list = [X_train[n] for n in X_train]
    y_list: list = [y_train[n] for n in y_train]

    node_w: torch.Tensor = compute_class_weights(y_list)
    edge_w: torch.Tensor = compute_edge_weights(y_list)
    print(f"Node class weights: {node_w.tolist()}")
    print(f"Edge class weights: {edge_w.tolist()}")

    # Reuses v11/v8b cache — same preprocessing
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
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
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
    trainer.fit(wrapper, loader, ckpt_path="/data/anhld48/phatnt/Graduation-Thesis-BsC-2026/src/lightning_logs/version_58/checkpoints/epoch=13-step=2576.ckpt")

    path: str = os.path.join(model_directory_path, "model.pt")
    sd: dict = wrapper.model.state_dict()
    torch.save(
        {k.replace("module.", ""): v for k, v in sd.items()},
        path,
    )
    print(f"Saved model to {path}")


# ============================================================
# Inference (identical pattern to v11)
# ============================================================
def infer_batch_local(
    dfs: list[pd.DataFrame], model: nn.Module,
    device: str = "cuda", batch_size: int = 64,
    cache_dir: str | None = None,
) -> list[pd.DataFrame]:
    cache_path = os.path.join(cache_dir, f"infer_v16_nk{N_KERNEL}.pkl") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            samples = pickle.load(f)
    else:
        import multiprocessing as mp
        n_workers = max(1, mp.cpu_count() - 1)
        args = [(df, None) for df in dfs]
        print(f"Building edge tensors ({len(dfs)} graphs, {n_workers} workers)...")
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            samples = list(tqdm(pool.map(_build_single, args, chunksize=8), total=len(args)))
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    dataset = InMemoryDataset(samples)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    results: list[pd.DataFrame | None] = [None] * len(dfs)
    idx_off: int = 0

    with torch.no_grad():
        for batch in loader:
            edge_logits, node_logits_list = model(
                batch["edge_data"].to(device),
                batch["edge_types"].to(device),
                batch["edge_mask"].to(device),
                batch["cols"],
                struct_rel=batch["struct_rel"].to(device),
            )
            B_curr: int = len(batch["cols"])
            for b in range(B_curr):
                cols: list[str] = batch["cols"][b]
                p: int = len(cols)

                # Build probability matrix from edge softmax
                emask: torch.Tensor = batch["edge_mask"][b]
                eprobs: np.ndarray = F.softmax(edge_logits[b][emask], dim=-1)[:, 1].cpu().numpy()
                pred_matrix: np.ndarray = np.zeros((p, p), dtype=np.float64)
                count: int = 0
                for ui in range(p):
                    for vi in range(p):
                        if ui != vi:
                            pred_matrix[ui, vi] = float(eprobs[count])
                            count += 1

                # Convert to DAG with cycle removal
                dag_matrix: np.ndarray = transform_proba_to_DAG(cols, pred_matrix)
                A: pd.DataFrame = pd.DataFrame(
                    dag_matrix.astype(int), index=cols, columns=cols,
                )
                results[idx_off + b] = A
            idx_off += B_curr
    return results


def infer_batch_adj_local(
    dfs: list[pd.DataFrame], model: nn.Module,
    device: str = "cuda", batch_size: int = 64,
    cache_dir: str | None = None,
) -> list[pd.DataFrame]:
    return infer_batch_local(dfs, model, device, batch_size, cache_dir)


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
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("v16 Edge-Pair Conv (Architecture A)")
    print(f"  v11 backbone + NodeConvBranch (32ch → {N_NODECONV_BLOCKS} blocks)")
    print(f"  {N_STRUCT_TYPES} struct types, {N_HEADS} heads, d={D_MODEL}")
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
    model = ADIAModel(d=D_MODEL, aug_noise_std=0.0)
    model.load_state_dict(torch.load("resources/model.pt", map_location=device, weights_only=True))
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
    print("\nPer-class accuracy:")
    accs = []
    for cls in CLASS_NAMES:
        n = ct[cls]
        acc = cc[cls] / n if n > 0 else 0.0
        accs.append(acc)
        print(f"  {cls:25s}: {acc:.4f}  (n={n})")
    print(f"\nBalanced Accuracy: {np.mean(accs):.4f}")
