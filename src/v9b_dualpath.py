"""
v9b_dualstream.py — ADIA Causal Discovery
  Dual-Stream Node-Centric Architecture

Architecture:
  1. For all K non-X/Y nodes, extract 4 edges each (v→X, v→Y, X→v, Y→v)
  2. Shared Conv backbone → (K*4, d) embeddings
  3. Reshape to (K, 4, d)

  Stream A (Node-Centric — per-node classification):
    + Role embeddings (4 learnable)
    → 2× Self-attention over 4 tokens per node
    → Mean pool → (K, d) node embeddings

  Stream B (Graph Context — zero extra conv cost):
    Pool conv embeddings by role across ALL nodes:
      X_in_ctx  = MeanPool of all X→v embeddings  → (d,)
      X_out_ctx = MeanPool of all v→X embeddings  → (d,)
      Y_in_ctx  = MeanPool of all Y→v embeddings  → (d,)
      Y_out_ctx = MeanPool of all v→Y embeddings  → (d,)
    → MLP([X_in, X_out, Y_in, Y_out]) → graph_context (d,)
    → Broadcast to all K nodes

  Fusion:
    concat(node_emb, graph_context) → MLP → 8-class

  Auxiliary (training only):
    Edge head: each of K*4 edges → Linear(d, 2) → binary existence

Same 8ch preprocessing (multi-bandwidth kernel + ANM residuals).
Go/no-go on base 25K. Target: beat v8b base 76.94%.

Usage:
    python v9b_dualstream.py
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
N_CHANNELS: int = 8
D_MODEL: int = 64
CLASS_NAMES: list[str] = [
    "Confounder", "Collider", "Mediator",
    "Cause of X", "Cause of Y",
    "Consequence of X", "Consequence of Y",
    "Independent",
]

BANDWIDTHS: list[float] = [0.2, 0.5, 1.0]

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


# ============================================================
# Data Preprocessing (same 8ch as v8b)
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

        # Full edge labels for auxiliary edge head
        edge_labels: list[int] = []
        for ui in range(p):
            for vi in range(p):
                if ui != vi:
                    edge_labels.append(int(adj_np[ui, vi]))
        result["edge_labels"] = np.array(edge_labels, dtype=np.int64)

        # Node labels
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
# Dataset
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
        if "node_labels" in item:
            result["node_labels"] = torch.from_numpy(item["node_labels"])
        return result


def collate_fn(batch: list[dict]) -> dict:
    max_E: int = max(item["edge_data"].shape[0] for item in batch)
    B: int = len(batch)

    edge_data: torch.Tensor = torch.zeros(B, max_E, N_CHANNELS, N_OBS)
    edge_types: torch.Tensor = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask: torch.Tensor = torch.zeros(B, max_E, dtype=torch.bool)
    edge_labels: torch.Tensor = torch.full((B, max_E), -1, dtype=torch.long)

    max_K: int = max(
        (item["node_labels"].shape[0] if "node_labels" in item else 0)
        for item in batch
    )
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
        if "edge_labels" in item:
            has_labels = True
            edge_labels[b, :E] = item["edge_labels"]
        if "node_labels" in item:
            has_labels = True
            K: int = item["node_labels"].shape[0]
            node_labels[b, :K] = item["node_labels"]
            node_mask[b, :K] = True

    out: dict = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "edge_mask": edge_mask,
        "cols": cols_list,
    }
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels
        out["node_mask"] = node_mask
    return out


# ============================================================
# Model Architecture — Dual-Stream Node-Centric
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, d: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(n_groups, d)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.conv(x)))


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


class SelfAttentionLayer(nn.Module):
    def __init__(self, d: int = 64, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class DualStreamModel(nn.Module):
    """
    Dual-Stream Node-Centric Causal Classifier.

    Stream A: Per-node, 4 edges → conv → role emb → self-attn → node emb
    Stream B: Pool conv embeddings by role across all nodes → graph context
    Fusion: concat(node_emb, graph_ctx) → MLP → 8-class
    Aux: Edge head on gathered edges (training only)
    """

    def __init__(
        self, d: int = None, aug_noise_std: float = 0.0,
    ):
        super().__init__()
        d = d or D_MODEL
        self.d: int = d
        self.aug_noise_std: float = aug_noise_std

        # Shared conv backbone
        self.extractor = EdgeFeatureExtractor(d, n_channels=N_CHANNELS)

        # Stream A: 4 role embeddings + merge + self-attention
        self.role_emb = nn.Embedding(4, d)  # 0=v→X, 1=v→Y, 2=X→v, 3=Y→v
        self.role_merge = nn.Sequential(
            nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU(),
        )
        self.node_attn = nn.ModuleList(
            [SelfAttentionLayer(d) for _ in range(2)]
        )

        # Stream B: graph context from pooled role embeddings
        # Input: 4 role-pooled vectors (each d) → context vector (d)
        self.context_proj = nn.Sequential(
            nn.Linear(4 * d, d), nn.LayerNorm(d), nn.GELU(),
        )

        # Fusion: node_emb (d) + graph_ctx (d) → 8-class
        self.classifier = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, N_CLASSES),
        )

        # Auxiliary edge head (training only)
        self.edge_head = nn.Linear(d, 2)

    def _eidx(self, u: int, v: int, p: int) -> int:
        return u * (p - 1) + v - (1 if v > u else 0)

    def forward(
        self,
        edge_data: torch.Tensor,    # (B, E, C, N)
        edge_types: torch.Tensor,    # (B, E)
        edge_mask: torch.Tensor,     # (B, E)
        cols_list: list[list[str]],
    ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        """
        Returns:
            node_logits_list: list of (K, 8) tensors per batch
            edge_logits_list: list of (K*4, 2) tensors per batch (for aux loss)
        """
        B: int
        E: int
        C: int
        N: int
        B, E, C, N = edge_data.shape

        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std

        device: torch.device = edge_data.device
        role_ids: torch.Tensor = torch.arange(4, device=device)
        role_embs: torch.Tensor = self.role_emb(role_ids)  # (4, d)

        node_logits_list: list[torch.Tensor | None] = []
        edge_logits_list: list[torch.Tensor | None] = []

        for b in range(B):
            cols: list[str] = cols_list[b]
            p: int = len(cols)
            col2idx: dict[str, int] = {c: i for i, c in enumerate(cols)}
            other: list[str] = [c for c in cols if c not in ("X", "Y")]
            K: int = len(other)

            if K == 0:
                node_logits_list.append(None)
                edge_logits_list.append(None)
                continue

            xi: int = col2idx["X"]
            yi: int = col2idx["Y"]

            # Gather 4 edges per node: v→X, v→Y, X→v, Y→v
            edge_indices: list[int] = []
            for v_name in other:
                vi: int = col2idx[v_name]
                edge_indices.extend([
                    self._eidx(vi, xi, p),  # role 0: v→X
                    self._eidx(vi, yi, p),  # role 1: v→Y
                    self._eidx(xi, vi, p),  # role 2: X→v
                    self._eidx(yi, vi, p),  # role 3: Y→v
                ])

            idx_tensor: torch.Tensor = torch.tensor(
                edge_indices, dtype=torch.long, device=device
            )
            gathered: torch.Tensor = edge_data[b, idx_tensor]  # (K*4, C, N)

            # Shared conv backbone
            conv_emb: torch.Tensor = self.extractor(gathered)  # (K*4, d)

            # Auxiliary edge head (on all K*4 edges)
            edge_logits: torch.Tensor = self.edge_head(conv_emb)  # (K*4, 2)
            edge_logits_list.append(edge_logits)

            conv_emb_4 = conv_emb.view(K, 4, self.d)  # (K, 4, d)

            # ---- Stream A: per-node self-attention ----
            role_expanded: torch.Tensor = role_embs.unsqueeze(0).expand(K, -1, -1)
            merged: torch.Tensor = self.role_merge(
                torch.cat([conv_emb_4, role_expanded], dim=-1)
            )  # (K, 4, d)

            for layer in self.node_attn:
                merged = layer(merged)  # (K, 4, d)

            node_emb: torch.Tensor = merged.mean(dim=1)  # (K, d)

            # ---- Stream B: graph context from role-pooled embeddings ----
            # conv_emb_4 is (K, 4, d) with roles [v→X, v→Y, X→v, Y→v]
            # Pool each role across all K nodes
            vX_pool: torch.Tensor = conv_emb_4[:, 0, :].mean(dim=0)  # (d,) — all v→X
            vY_pool: torch.Tensor = conv_emb_4[:, 1, :].mean(dim=0)  # (d,) — all v→Y
            Xv_pool: torch.Tensor = conv_emb_4[:, 2, :].mean(dim=0)  # (d,) — all X→v
            Yv_pool: torch.Tensor = conv_emb_4[:, 3, :].mean(dim=0)  # (d,) — all Y→v

            graph_ctx: torch.Tensor = self.context_proj(
                torch.cat([vX_pool, vY_pool, Xv_pool, Yv_pool], dim=-1)
            )  # (d,)

            # ---- Fusion ----
            graph_ctx_expanded: torch.Tensor = graph_ctx.unsqueeze(0).expand(K, -1)
            fused: torch.Tensor = torch.cat([node_emb, graph_ctx_expanded], dim=-1)
            node_logits: torch.Tensor = self.classifier(fused)  # (K, 8)

            node_logits_list.append(node_logits)

        return node_logits_list, edge_logits_list


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


class DualStreamWrapper(pl.LightningModule):
    def __init__(
        self, d: int = None, node_class_weights: torch.Tensor = None,
        edge_class_weights: torch.Tensor = None,
        lr: float = 1e-3, max_epochs: int = 20,
        aug_noise_std: float = 0.0,
    ):
        super().__init__()
        d = d or D_MODEL
        self.model = DualStreamModel(d, aug_noise_std=aug_noise_std)
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
        )

    def _compute_loss(self, batch: dict, split: str) -> torch.Tensor:
        node_logits_list: list
        edge_logits_list: list
        node_logits_list, edge_logits_list = self(batch)
        B: int = len(node_logits_list)

        total_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
        n_terms: int = 0

        # Auxiliary edge loss on gathered 4 edges per node
        if "edge_labels" in batch:
            all_edge_logits: list[torch.Tensor] = []
            all_edge_labels: list[torch.Tensor] = []
            el: torch.Tensor = batch["edge_labels"].to(self.device)

            for b in range(B):
                if edge_logits_list[b] is None:
                    continue
                cols: list[str] = batch["cols"][b]
                p: int = len(cols)
                col2idx: dict[str, int] = {c: i for i, c in enumerate(cols)}
                other: list[str] = [c for c in cols if c not in ("X", "Y")]
                xi: int = col2idx["X"]
                yi: int = col2idx["Y"]

                # Gather the same 4 edge labels per node
                label_indices: list[int] = []
                for v_name in other:
                    vi: int = col2idx[v_name]
                    eidx = self.model._eidx
                    label_indices.extend([
                        eidx(vi, xi, p), eidx(vi, yi, p),
                        eidx(xi, vi, p), eidx(yi, vi, p),
                    ])
                idx_t: torch.Tensor = torch.tensor(
                    label_indices, dtype=torch.long, device=self.device
                )
                all_edge_logits.append(edge_logits_list[b])
                all_edge_labels.append(el[b, idx_t])

            if all_edge_logits:
                cat_el: torch.Tensor = torch.cat(all_edge_logits, dim=0)
                cat_ll: torch.Tensor = torch.cat(all_edge_labels, dim=0)
                valid_e: torch.Tensor = cat_ll >= 0
                if valid_e.any():
                    edge_loss: torch.Tensor = self.edge_criterion(
                        cat_el[valid_e], cat_ll[valid_e]
                    )
                    total_loss = total_loss + edge_loss
                    n_terms += 1
                    self.log(f"{split}/edge_loss", edge_loss, prog_bar=False, sync_dist=True)

        # Node classification loss
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
                cat_nl: torch.Tensor = torch.cat(all_node_logits, dim=0)
                cat_nll: torch.Tensor = torch.cat(all_node_labels, dim=0)
                valid_n: torch.Tensor = cat_nll >= 0
                if valid_n.any():
                    node_loss: torch.Tensor = self.node_criterion(
                        cat_nl[valid_n], cat_nll[valid_n]
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

    # Class weights (reuse v8b)
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
        print(f"Saving dataset cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    dataset = InMemoryDataset(samples)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} samples")

    wrapper = DualStreamWrapper(
        d=D_MODEL, node_class_weights=node_w, edge_class_weights=edge_w,
        lr=LR, max_epochs=MAX_EPOCHS, aug_noise_std=AUG_NOISE_STD,
    )
    n_params: int = sum(p.numel() for p in wrapper.model.parameters())
    print(f"DualStreamModel params: {n_params:,}")

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

    results = [None] * len(all_items)
    idx_off = 0
    for batch in tqdm(loader, desc="inferring"):
        node_logits_list, _ = model(
            batch["edge_data"].to(device), batch["edge_types"].to(device),
            batch["edge_mask"].to(device), batch["cols"])
        for b in range(len(batch["cols"])):
            item = all_items[idx_off + b]
            cols, p = item["cols"], item["p"]

            adj = np.zeros((p, p), dtype=int)
            xi_idx = cols.index("X")
            yi_idx = cols.index("Y")
            adj[xi_idx, yi_idx] = 1

            if node_logits_list[b] is not None:
                other_nodes = [n for n in cols if n not in ("X", "Y")]
                node_preds = torch.argmax(node_logits_list[b], dim=-1)
                for k, nn_ in enumerate(other_nodes):
                    for (s, d) in patterns[CLASS_NAMES[node_preds[k].item()]](nn_):
                        si = cols.index(s)
                        di = cols.index(d)
                        adj[si, di] = 1

            A = pd.DataFrame(adj, columns=cols, index=cols)
            results[idx_off + b] = A
        idx_off += len(batch["cols"])
    return results


def infer(X_test, model_directory_path, id_column_name, prediction_column_name):
    path = os.path.join(model_directory_path, "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DualStreamModel(d=D_MODEL, aug_noise_std=0.0)
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
    print("v9b Dual-Stream: Node-centric + Graph Context + Edge Aux")
    print(f"Channels: {N_CHANNELS}, D_MODEL: {D_MODEL}")
    print(f"Config: bs={BATCH_SIZE}, lr={LR}, epochs={MAX_EPOCHS}")
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
    model = DualStreamModel(d=D_MODEL, aug_noise_std=0.0)
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
