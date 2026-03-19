"""
v8b_anm.py — ADIA Causal Discovery
  Multi-bandwidth kernel regression (5ch) + ANM multivariate residuals (3ch)
  = 8 input channels per edge

Channels per edge (u→v):
  Ch 0:  sorted u observations
  Ch 1:  v observations sorted by u
  Ch 2:  kernel regression coeff at bandwidth=0.2
  Ch 3:  kernel regression coeff at bandwidth=0.5
  Ch 4:  kernel regression coeff at bandwidth=1.0
  Ch 5:  ANM multivariate residual (v - ŷ_v) at bandwidth=0.2, sorted by u
  Ch 6:  ANM multivariate residual (v - ŷ_v) at bandwidth=0.5, sorted by u
  Ch 7:  ANM multivariate residual (v - ŷ_v) at bandwidth=1.0, sorted by u

Architecture: v2 baseline with N_CHANNELS=8, no StatProjector, no focal loss.
Training: base 25K, bs=16, lr=1e-3, 30 epochs, 2 GPU DDP.

Usage:
    python v8b_anm.py
"""

# @crunch/keep:on
import crunch

import typing
import os
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

# === Training config ===
MAX_EPOCHS: int = 30
BATCH_SIZE: int = 16
LR: float = 1e-3
AUG_NOISE_STD: float = 0.01
N_AUG: int = 1
LOCAL_CACHE_DIR: str = "dataset_cache/"
IS_CLOUD_SUBMIT: bool = False
SHARD_SIZE: int = 50_000


# ============================================================
# Graph Utilities (unchanged)
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
# Data Preprocessing
# ============================================================
def _edge_type(u_name: str, v_name: str) -> int:
    """Edge type encoding (7 types)."""
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
    """
    Multivariate kernel regression for ALL target variables at once.
    Uses pre-selected subsample indices for consistency across bandwidths.

    Returns:
        coeff_map: dict mapping (k, j) -> np.ndarray of shape (N,)
        resid_map: dict mapping j -> np.ndarray of shape (N,)
            residual[i] = x_j[i] - ŷ_j(i)  where ŷ uses all other variables
    """
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

        # Extract per-predictor coefficients (for conv channels)
        for idx_in_other, k in enumerate(other_cols):
            coeff_sub: np.ndarray = c_all[:, idx_in_other + 1]
            coeff_map[(k, j)] = coeff_sub[nearest].astype(np.float32)

        # Compute residuals: x_j - ŷ_j using NN-interpolated coefficients
        # ŷ_j(i) = c_0[NN(i)] + Σ_k c_k[NN(i)] * x_k(i)
        c_nn: np.ndarray = c_all[nearest]  # (N, p) — all coeffs for target j
        # Build predictor matrix for all N points
        X_full: np.ndarray = np.concatenate(
            [np.ones((N, 1)), data[:, other_cols]], axis=1
        )  # (N, p)
        y_hat: np.ndarray = np.sum(c_nn * X_full, axis=1)  # (N,)
        resid_map[j] = (data[:, j] - y_hat).astype(np.float32)

    return coeff_map, resid_map


def build_edge_tensor(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 8-channel edge tensor: multi-bandwidth kernel + ANM residuals.

    Channels per edge (u→v):
      Ch 0:  sorted u observations
      Ch 1:  v observations sorted by u
      Ch 2-4:  kernel regression coeff at bw 0.2, 0.5, 1.0
      Ch 5-7:  ANM multivariate residual (v - ŷ_v) at bw 0.2, 0.5, 1.0

    Returns:
      edge_data:   (E, N_CHANNELS, N) float32
      edge_types:  (E,) int64
    """
    cols: list[str] = list(df.columns)
    p: int = len(cols)
    data: np.ndarray = df.values.astype(np.float32)
    N: int = data.shape[0]

    # --- Multi-bandwidth kernel regression (coefficients + residuals) ---
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

    # --- Build edge tensors ---
    edges: list[np.ndarray] = []
    edge_types: list[int] = []

    for i, u_name in enumerate(cols):
        sort_idx: np.ndarray = np.argsort(data[:, i])
        u_sorted: np.ndarray = data[sort_idx, i]

        for j, v_name in enumerate(cols):
            if i == j:
                continue

            v_sorted_by_u: np.ndarray = data[sort_idx, j]

            # Kernel regression channels (sorted by u)
            kernel_channels: list[np.ndarray] = [
                cm[(i, j)][sort_idx] for cm in coeff_maps
            ]

            # ANM residual channels (sorted by u)
            anm_channels: list[np.ndarray] = [
                rm[j][sort_idx] for rm in resid_maps
            ]

            # Stack all 8 channels
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
# Sample Builders
# ============================================================
def _build_single(args: tuple) -> dict:
    """Worker: builds edge tensor + precomputes all labels for one sample."""
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


def _remap_xy_names(
    cols: list[str], new_x: str, new_y: str
) -> dict[str, str]:
    result: list[str] = list(cols)
    pos: dict[str, int] = {c: i for i, c in enumerate(cols)}
    i_nx: int = pos[new_x]
    i_x: int = pos["X"]
    result[i_nx], result[i_x] = result[i_x], result[i_nx]
    pos2: dict[str, int] = {c: i for i, c in enumerate(result)}
    i_ny: int = pos2[new_y]
    i_y: int = pos2["Y"]
    result[i_ny], result[i_y] = result[i_y], result[i_ny]
    return {cols[i]: result[i] for i in range(len(cols))}


def _build_augmented(args: tuple) -> list[dict]:
    df: pd.DataFrame
    y_df: pd.DataFrame | None
    df, y_df, _ = args

    results: list[dict] = [_build_single((df, y_df))]
    if y_df is None:
        return results

    cols: list[str] = list(df.columns)
    adj_np: np.ndarray = y_df.values
    col_idx: dict[str, int] = {c: i for i, c in enumerate(y_df.columns)}

    for a_name in y_df.columns:
        for b_name in y_df.columns:
            if a_name == b_name:
                continue
            if adj_np[col_idx[a_name], col_idx[b_name]] != 1:
                continue
            if a_name == "X" and b_name == "Y":
                continue
            rename_map: dict[str, str] = _remap_xy_names(cols, a_name, b_name)
            results.append(_build_single((
                df.rename(columns=rename_map),
                y_df.rename(index=rename_map, columns=rename_map),
            )))

    return results


# ============================================================
# Sharded Dataset + Sampler (for future XY aug runs)
# ============================================================
def _save_sharded(
    samples: list[dict], shard_dir: str
) -> dict:
    os.makedirs(shard_dir, exist_ok=True)
    n_shards: int = (len(samples) + SHARD_SIZE - 1) // SHARD_SIZE
    shard_sizes: list[int] = []
    for i in range(n_shards):
        start: int = i * SHARD_SIZE
        end: int = min((i + 1) * SHARD_SIZE, len(samples))
        with open(os.path.join(shard_dir, f"shard_{i:04d}.pkl"), "wb") as f:
            pickle.dump(samples[start:end], f, protocol=pickle.HIGHEST_PROTOCOL)
        shard_sizes.append(end - start)
        print(f"  Saved shard {i+1}/{n_shards}: {end-start} samples")
    meta: dict = {
        "n_samples": len(samples),
        "n_shards": n_shards,
        "shard_sizes": shard_sizes,
    }
    with open(os.path.join(shard_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved {len(samples)} samples in {n_shards} shards.")
    return meta


def _build_and_save_shards(
    X_list: list, y_list: list | None,
    shard_dir: str, n_workers: int = 10, augment: bool = False,
) -> dict:
    use_aug: bool = augment and y_list is not None
    if use_aug:
        args: list[tuple] = [
            (X_list[i], y_list[i], 0) for i in range(len(X_list))
        ]
        print(f"Building augmented dataset ({len(args)} base, X/Y remap, {n_workers} workers)...")
        ctx = __import__('multiprocessing').get_context('fork')
        if n_workers > 1:
            raw_nested: list = [None] * len(args)
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {
                    pool.submit(_build_augmented, a): idx
                    for idx, a in enumerate(args)
                }
                for fut in tqdm(as_completed(futures), total=len(args)):
                    raw_nested[futures[fut]] = fut.result()
        else:
            raw_nested = [_build_augmented(a) for a in tqdm(args)]
        raw: list[dict] = []
        for group in raw_nested:
            raw.extend(group)
        print(f"Total samples: {len(raw)} ({len(raw)/len(args):.1f}x avg)")
    else:
        args = [
            (X_list[i], y_list[i] if y_list else None)
            for i in range(len(X_list))
        ]
        print(f"Building dataset ({len(args)} samples, {n_workers} workers)...")
        ctx = __import__('multiprocessing').get_context('fork')
        if n_workers > 1:
            raw = [None] * len(args)
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {
                    pool.submit(_build_single, a): idx
                    for idx, a in enumerate(args)
                }
                for fut in tqdm(as_completed(futures), total=len(args)):
                    raw[futures[fut]] = fut.result()
        else:
            raw = [_build_single(a) for a in tqdm(args)]

    meta: dict = _save_sharded(raw, shard_dir)
    del raw
    gc.collect()
    return meta


class ShardGroupedSampler(Sampler):
    """Yields indices grouped by shard. Shuffles shard order + within-shard each epoch."""

    def __init__(self, shard_sizes: list[int], seed: int = 42):
        self.shard_sizes: list[int] = shard_sizes
        self.n_shards: int = len(shard_sizes)
        self.total: int = sum(shard_sizes)
        self.shard_offsets: list[int] = []
        offset: int = 0
        for s in shard_sizes:
            self.shard_offsets.append(offset)
            offset += s
        self.epoch: int = 0
        self.seed: int = seed

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        shard_order: np.ndarray = rng.permutation(self.n_shards)
        indices: list[int] = []
        for si in shard_order:
            local: np.ndarray = (
                rng.permutation(self.shard_sizes[si]) + self.shard_offsets[si]
            )
            indices.extend(local.tolist())
        return iter(indices)

    def __len__(self) -> int:
        return self.total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class CausalEdgeDataset(Dataset):
    """Shard-based dataset. Only one shard loaded at a time."""

    def __init__(self, shard_dir: str):
        with open(os.path.join(shard_dir, "meta.pkl"), "rb") as f:
            meta: dict = pickle.load(f)
        self.shard_dir: str = shard_dir
        self.n_samples: int = meta["n_samples"]
        self.n_shards: int = meta["n_shards"]
        self.shard_sizes: list[int] = meta["shard_sizes"]
        self._loaded_shard: int = -1
        self._shard_data: list | None = None

    def _find_shard(self, idx: int) -> tuple[int, int]:
        offset: int = 0
        for si, sz in enumerate(self.shard_sizes):
            if idx < offset + sz:
                return si, idx - offset
            offset += sz
        raise IndexError(f"Index {idx} out of range")

    def _load_shard(self, shard_idx: int) -> None:
        if self._loaded_shard == shard_idx:
            return
        path: str = os.path.join(
            self.shard_dir, f"shard_{shard_idx:04d}.pkl"
        )
        with open(path, "rb") as f:
            self._shard_data = pickle.load(f)
        self._loaded_shard = shard_idx

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        shard_idx: int
        local_idx: int
        shard_idx, local_idx = self._find_shard(idx)
        self._load_shard(shard_idx)
        item: dict = self._shard_data[local_idx]
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


class InMemoryDataset(Dataset):
    """Simple in-memory dataset for base training (no sharding)."""

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
        "cols": cols_list,
    }
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels
        out["node_mask"] = node_mask
    return out


# ============================================================
# Model Architecture — v2 baseline with N_CHANNELS=8
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


class ADIAModel(nn.Module):
    """
    v8b: v2 baseline architecture with 8-channel input (multibw + ANM).
    Pipeline: EdgeFeatureExtractor(11ch) → Merge(conv+type) → 2×SelfAttn → heads.
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
        self.attn_layers = nn.ModuleList(
            [SelfAttentionLayer(d) for _ in range(2)]
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
    ) -> tuple:
        B: int
        E: int
        C: int
        N: int
        B, E, C, N = edge_data.shape

        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std

        x_flat: torch.Tensor = edge_data.view(B * E, C, N)
        conv_emb: torch.Tensor = self.extractor(x_flat).view(B, E, self.d)
        type_emb: torch.Tensor = self.edge_type_emb(edge_types)
        edge_emb: torch.Tensor = self.edge_merge([conv_emb, type_emb])

        inv_mask: torch.Tensor = ~edge_mask
        for layer in self.attn_layers:
            edge_emb = layer(edge_emb, key_padding_mask=inv_mask)

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
        )

    def _compute_loss(self, batch: dict, split: str) -> torch.Tensor:
        edge_logits: torch.Tensor
        node_logits_list: list
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

    # Class weights
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

    # Build dataset (in-memory, base 25K)
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
        print(f"Cached {len(samples)} samples.")

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
    print(f"Params: {sum(p.numel() for p in wrapper.model.parameters()):,}")

    trainer = pl.Trainer(
        accelerator="gpu", devices=2, strategy="ddp",
        max_epochs=MAX_EPOCHS, precision="32-true",
        use_distributed_sampler=True,
        logger=True, enable_checkpointing=True, enable_progress_bar=True,
    )
    trainer.fit(wrapper, loader)

    # Save model
    path: str = os.path.join(model_directory_path, "model.pt")
    sd: dict = wrapper.model.state_dict()
    torch.save(
        {k.replace("module.", ""): v for k, v in sd.items()}, path
    )
    print(f"Model saved to {path}")


# ============================================================
# Inference
# ============================================================
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
    for batch in tqdm(loader, desc="infering"):
        edge_logits, node_logits_list = model(
            batch["edge_data"].to(device), batch["edge_types"].to(device),
            batch["edge_mask"].to(device), batch["cols"])
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
# Main — local training + evaluation
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("v8b_anm: Multi-bandwidth + ANM residuals (no CMI)")
    print(f"Channels: {N_CHANNELS} (2 sorted + {len(BANDWIDTHS)} kernel + {len(BANDWIDTHS)} ANM)")
    print(f"Config: bs={BATCH_SIZE}, lr={LR}, epochs={MAX_EPOCHS}")
    print("=" * 60)

    X_train = pd.read_pickle("data/X_train.pickle")
    y_train = pd.read_pickle("data/y_train.pickle")
    print(f"Loaded {len(X_train)} training samples.")
    train(X_train, y_train, model_directory_path="resources")

    # Only rank 0 runs evaluation (DDP spawns 2 processes that both hit __main__)
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