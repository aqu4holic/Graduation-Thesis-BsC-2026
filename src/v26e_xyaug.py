"""
v26e_xyaug.py — ADIA Causal Discovery
  v26e (12ch density, 80.47% BA) + XY remap augmentation (~11x data)

Same architecture as v26e:
  12-channel node images (4 raw + 4 ANM + 4 kernel coeff) fused with edge embeddings.
  Edge pipeline as context encoder with lambda-weighted auxiliary loss.

XY aug optimization: kernel regression (coeff_map, resid_map) and edge_data
computed ONCE per graph. For each XY remap, only rebuild:
  - edge_types (cheap: just name swaps)
  - node_images (cheap: histograms from precomputed coeff/resid maps)
  - labels (cheap: adjacency relabeling)

Shard-based dataset: ~263K augmented samples, streamed to disk in 50K shards.

Usage:
    python v26e_xyaug.py
"""

# @crunch/keep:on
import crunch

import os
import glob
import typing
import gc
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers.wandb import WandbLogger
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
N_NODE_CHANNELS: int = 8
SCATTER_SIGMA: float = 4.0
ANM_SIGMA: float = 2.0
ANM_BW: float = 0.5
LOSS_LAMBDA: float = 0.7  # total_loss = lambda * edge_loss + (1 - lambda) * node_loss
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

BANDWIDTHS: list[float] = [0.2, 0.5, 1.0]
N_CHANNELS_1D: int = 2 + len(BANDWIDTHS) + len(BANDWIDTHS)  # 8

MAX_EPOCHS: int = 10
BATCH_SIZE: int = 64
LR: float = 1e-3
AUG_NOISE_STD: float = 0.01
LOCAL_CACHE_DIR: str = "dataset_cache/"
SHARD_SIZE: int = 50_000
IS_CLOUD_SUBMIT: bool = False
VERSION: str = "v26e_xyaug"
VERSION_NAME: str = f"{VERSION}_noch{N_NODE_CHANNELS}_ss{SCATTER_SIGMA}_as{ANM_SIGMA}"
MODEL_NAME: str = f"{VERSION_NAME}_lam{LOSS_LAMBDA}_lr{LR}_aug{AUG_NOISE_STD}"


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
# 1D Preprocessing (for edge pipeline)
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


def build_edge_tensor(df):
    cols = list(df.columns)
    p = len(cols)
    data = df.values.astype(np.float32)
    N = data.shape[0]

    n_sub = min(N_KERNEL, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)

    coeff_maps, resid_maps = [], []
    for bw in BANDWIDTHS:
        cm, rm = compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=bw)
        coeff_maps.append(cm)
        resid_maps.append(rm)

    edges, edge_types = [], []
    for i, u_name in enumerate(cols):
        sort_idx = np.argsort(data[:, i])
        u_sorted = data[sort_idx, i]
        for j, v_name in enumerate(cols):
            if i == j:
                continue
            v_sorted_by_u = data[sort_idx, j]
            kernel_ch = [cm[(i, j)][sort_idx] for cm in coeff_maps]
            anm_ch = [rm[j][sort_idx] for rm in resid_maps]
            channels = [u_sorted, v_sorted_by_u] + kernel_ch + anm_ch
            edges.append(np.stack(channels, axis=0))
            edge_types.append(_edge_type(u_name, v_name))

    return np.stack(edges, axis=0).astype(np.float32), np.array(edge_types, dtype=np.int64)


# ============================================================
# 2D Preprocessing — 12-channel scatter density images
# ============================================================
def build_scatter_density(source, target, grid_size=32, sigma=5.0):
    """Raw [-1,1] range, Gaussian smoothed, NO density normalization."""
    hist, _, _ = np.histogram2d(
        source, target, bins=grid_size, range=[[-1, 1], [-1, 1]]
    )
    hist_smooth = gaussian_filter(hist.astype(np.float32), sigma=sigma)
    return hist_smooth


def build_node_images(df, coeff_map=None, resid_map=None):
    """Build 12-channel scatter density images for each non-X/Y node.

    ch0-3:  raw joint density (sigma=SCATTER_SIGMA)
    ch4-7:  ANM residual density (sigma=ANM_SIGMA)
    ch8-11: kernel coefficient density (sigma=ANM_SIGMA)
    """
    cols = list(df.columns)
    data = df.values.astype(np.float32)
    col2idx = {c: i for i, c in enumerate(cols)}
    xi = col2idx["X"]
    yi = col2idx["Y"]
    other_nodes = [c for c in cols if c not in ("X", "Y")]

    x_data = data[:, xi]
    y_data = data[:, yi]

    if coeff_map is None or resid_map is None:
        N = data.shape[0]
        n_sub = min(N_KERNEL, N)
        sub_idx = np.random.choice(N, n_sub, replace=False)
        coeff_map, resid_map = compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=ANM_BW)

    resid_X = resid_map[xi]
    resid_Y = resid_map[yi]

    node_images = {}
    for v_name in other_nodes:
        vi = col2idx[v_name]
        v_data = data[:, vi]
        resid_v = resid_map[vi]

        # Raw joint density channels
        chs = []
        chs.append(build_scatter_density(v_data, x_data, GRID_SIZE, sigma=SCATTER_SIGMA))
        chs.append(build_scatter_density(v_data, y_data, GRID_SIZE, sigma=SCATTER_SIGMA))
        chs.append(build_scatter_density(x_data, v_data, GRID_SIZE, sigma=SCATTER_SIGMA))
        chs.append(build_scatter_density(y_data, v_data, GRID_SIZE, sigma=SCATTER_SIGMA))

        # ANM residual density channels
        chs.append(build_scatter_density(v_data, resid_X, GRID_SIZE, sigma=ANM_SIGMA))
        chs.append(build_scatter_density(x_data, resid_v, GRID_SIZE, sigma=ANM_SIGMA))
        chs.append(build_scatter_density(v_data, resid_Y, GRID_SIZE, sigma=ANM_SIGMA))
        chs.append(build_scatter_density(y_data, resid_v, GRID_SIZE, sigma=ANM_SIGMA))

        if (N_NODE_CHANNELS > 8):
            # Kernel coefficient density channels
            coeff_v_to_x = coeff_map[(vi, xi)]
            coeff_x_to_v = coeff_map[(xi, vi)]
            coeff_v_to_y = coeff_map[(vi, yi)]
            coeff_y_to_v = coeff_map[(yi, vi)]

            chs.append(build_scatter_density(v_data, coeff_v_to_x, GRID_SIZE, sigma=ANM_SIGMA))
            chs.append(build_scatter_density(x_data, coeff_x_to_v, GRID_SIZE, sigma=ANM_SIGMA))
            chs.append(build_scatter_density(v_data, coeff_v_to_y, GRID_SIZE, sigma=ANM_SIGMA))
            chs.append(build_scatter_density(y_data, coeff_y_to_v, GRID_SIZE, sigma=ANM_SIGMA))

        # node_images[v_name] = np.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11], axis=0)
        node_images[v_name] = np.stack(chs, axis=0)

    return node_images


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
# Sample Builder (single, no aug)
# ============================================================
def _build_single(args):
    df, y_df = args
    data = df.values.astype(np.float32)
    N = data.shape[0]
    n_sub = min(N_KERNEL, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)

    coeff_map, resid_map = compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=ANM_BW)

    edge_data, edge_types = build_edge_tensor(df)
    node_images = build_node_images(df, coeff_map=coeff_map, resid_map=resid_map)

    cols = list(df.columns)
    p = len(cols)
    other_nodes = [c for c in cols if c not in ("X", "Y")]

    result = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "cols": cols,
        "p": p,
        "node_images": node_images,
        "other_nodes": other_nodes,
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
        node_labels_dict = get_labels(df_adj, adjacency_label)
        node_labels = [CLASS_NAMES.index(node_labels_dict[n]) for n in other_nodes]
        result["node_labels"] = np.array(node_labels, dtype=np.int64)

    return result


# ============================================================
# XY Remap Augmentation
# ============================================================
def _remap_xy_names(cols, new_x, new_y):
    result = list(cols)
    pos = {c: i for i, c in enumerate(cols)}
    i_nx, i_x = pos[new_x], pos["X"]
    result[i_nx], result[i_x] = result[i_x], result[i_nx]
    pos2 = {c: i for i, c in enumerate(result)}
    i_ny, i_y = pos2[new_y], pos2["Y"]
    result[i_ny], result[i_y] = result[i_y], result[i_ny]
    return {cols[i]: result[i] for i in range(len(cols))}


def _build_all_from_one_graph(args):
    """Build base + all XY-augmented samples from one graph.

    Optimization: edge_data, coeff_map, resid_map computed ONCE.
    For each remap: rebuild edge_types + node_images + labels (all cheap).
    """
    df, y_df = args

    # === Expensive part: compute ONCE ===
    edge_data, _ = build_edge_tensor(df)

    data = df.values.astype(np.float32)
    N = data.shape[0]
    n_sub = min(N_KERNEL, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)
    coeff_map, resid_map = compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=ANM_BW)

    cols = list(df.columns)
    p = len(cols)

    def _make_sample(cur_cols, cur_y_df):
        """Build sample reusing precomputed edge_data, coeff_map, resid_map."""
        # Edge types (cheap: just name-based)
        edge_types = []
        for i, u_name in enumerate(cur_cols):
            for j, v_name in enumerate(cur_cols):
                if i != j:
                    edge_types.append(_edge_type(u_name, v_name))
        edge_types = np.array(edge_types, dtype=np.int64)

        # Node images (cheap: histograms from precomputed maps)
        # Create a temporary df with remapped column names but same data
        tmp_df = pd.DataFrame(data, columns=cur_cols)
        node_images = build_node_images(tmp_df, coeff_map=coeff_map, resid_map=resid_map)

        other_nodes = [c for c in cur_cols if c not in ("X", "Y")]
        result = {
            "edge_data": edge_data,
            "edge_types": edge_types,
            "cols": cur_cols,
            "p": p,
            "node_images": node_images,
            "other_nodes": other_nodes,
        }

        if cur_y_df is not None:
            adj_np = cur_y_df.values.astype(np.float32)
            adj_cols = list(cur_y_df.columns)

            edge_labels = []
            for ui in range(p):
                for vi in range(p):
                    if ui != vi:
                        edge_labels.append(int(adj_np[ui, vi]))
            result["edge_labels"] = np.array(edge_labels, dtype=np.int64)

            _, adjacency_label = create_graph_label()
            df_adj = pd.DataFrame(adj_np, columns=adj_cols, index=adj_cols)
            try:
                node_labels_dict = get_labels(df_adj, adjacency_label)
            except KeyError:
                return None
            node_labels = [CLASS_NAMES.index(node_labels_dict[n]) for n in other_nodes]
            result["node_labels"] = np.array(node_labels, dtype=np.int64)

        return result

    results = []

    # Base sample (original X, Y)
    base = _make_sample(cols, y_df)
    if base is not None:
        results.append(base)

    # Augmented samples: for each directed edge A→B in ground truth, remap A→X, B→Y
    if y_df is not None:
        adj_np = y_df.values.astype(np.float32)
        col_idx = {c: i for i, c in enumerate(cols)}
        for a_name in y_df.columns:
            for b_name in y_df.columns:
                if a_name == b_name:
                    continue
                if adj_np[col_idx[a_name], col_idx[b_name]] != 1:
                    continue
                if a_name == "X" and b_name == "Y":
                    continue
                rename_map = _remap_xy_names(cols, a_name, b_name)
                new_cols = [rename_map[c] for c in cols]
                new_y_df = y_df.rename(index=rename_map, columns=rename_map)
                sample = _make_sample(new_cols, new_y_df)
                if sample is not None:
                    results.append(sample)

    return results


# ============================================================
# Sharded Dataset + Sampler
# ============================================================
def _flush_shard(buffer, shard_dir, shard_idx):
    path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    n = len(buffer)
    print(f"  Flushed shard {shard_idx}: {n} samples")
    return n


def _build_and_save_shards(X_list, y_list, shard_dir, n_workers=47):
    os.makedirs(shard_dir, exist_ok=True)

    buffer = []
    shard_sizes = []
    shard_idx = 0
    n_collected = 0

    import multiprocessing as mp
    ctx = mp.get_context('fork')

    args = [(X_list[i], y_list[i]) for i in range(len(X_list))]
    n_base = len(args)
    print(f"Building augmented dataset ({n_base} base graphs, {n_workers} workers)...")

    with ctx.Pool(processes=n_workers) as pool:
        for result_list in tqdm(
            pool.imap_unordered(_build_all_from_one_graph, args, chunksize=4),
            total=n_base,
        ):
            for sample in result_list:
                buffer.append(sample)
                n_collected += 1

            if len(buffer) >= SHARD_SIZE:
                shard_sizes.append(_flush_shard(buffer, shard_dir, shard_idx))
                shard_idx += 1
                buffer.clear()
                gc.collect()

    if buffer:
        shard_sizes.append(_flush_shard(buffer, shard_dir, shard_idx))
        buffer.clear()
        gc.collect()

    meta = {
        "n_samples": n_collected,
        "n_shards": len(shard_sizes),
        "shard_sizes": shard_sizes,
    }
    with open(os.path.join(shard_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"Done: {n_collected} samples in {len(shard_sizes)} shards.")
    return meta


class ShardGroupedSampler(Sampler):
    def __init__(self, shard_sizes, seed=42):
        self.shard_sizes = shard_sizes
        self.n_shards = len(shard_sizes)
        self.total = sum(shard_sizes)
        self.shard_offsets = []
        offset = 0
        for s in shard_sizes:
            self.shard_offsets.append(offset)
            offset += s
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        shard_order = rng.permutation(self.n_shards)
        indices = []
        for si in shard_order:
            local = rng.permutation(self.shard_sizes[si]) + self.shard_offsets[si]
            indices.extend(local.tolist())
        return iter(indices)

    def __len__(self):
        return self.total

    def set_epoch(self, epoch):
        self.epoch = epoch


class CausalEdgeDataset(Dataset):
    def __init__(self, shard_dir):
        meta_path = os.path.join(shard_dir, "meta.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.n_samples = meta["n_samples"]
        self.n_shards = meta["n_shards"]
        self.shard_sizes = meta["shard_sizes"]
        self.shard_dir = shard_dir
        self.shard_offsets = []
        offset = 0
        for s in self.shard_sizes:
            self.shard_offsets.append(offset)
            offset += s
        self._cur_shard_idx = -1
        self._cur_shard_data = None

    def __len__(self):
        return self.n_samples

    def _find_shard(self, idx):
        for si in range(self.n_shards):
            if idx < self.shard_offsets[si] + self.shard_sizes[si]:
                return si, idx - self.shard_offsets[si]
        raise IndexError(f"Index {idx} out of range")

    def _load_shard(self, si):
        if si != self._cur_shard_idx:
            path = os.path.join(self.shard_dir, f"shard_{si:04d}.pkl")
            with open(path, "rb") as f:
                self._cur_shard_data = pickle.load(f)
            self._cur_shard_idx = si

    def __getitem__(self, idx):
        si, local = self._find_shard(idx)
        self._load_shard(si)
        item = self._cur_shard_data[local]
        result = {
            "edge_data": torch.from_numpy(item["edge_data"]),
            "edge_types": torch.from_numpy(item["edge_types"]),
            "cols": item["cols"],
            "p": item["p"],
            "other_nodes": item["other_nodes"],
            "node_images": {k: torch.from_numpy(v) for k, v in item["node_images"].items()},
        }
        if "edge_labels" in item:
            result["edge_labels"] = torch.from_numpy(item["edge_labels"])
            result["node_labels"] = torch.from_numpy(item["node_labels"])
        return result


# ============================================================
# Collate
# ============================================================
def collate_fn(batch):
    max_E = max(item["edge_data"].shape[0] for item in batch)
    B = len(batch)

    edge_data = torch.zeros(B, max_E, N_CHANNELS_1D, N_OBS)
    edge_types = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask = torch.zeros(B, max_E, dtype=torch.bool)
    struct_rel = torch.full((B, max_E, max_E), 5, dtype=torch.long)

    max_K = max(len(item["other_nodes"]) for item in batch)
    node_imgs = torch.zeros(B, max_K, N_NODE_CHANNELS, GRID_SIZE, GRID_SIZE)
    node_img_mask = torch.zeros(B, max_K, dtype=torch.bool)

    edge_labels = torch.full((B, max_E), -1, dtype=torch.long)
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

        other = item["other_nodes"]
        for k, v_name in enumerate(other):
            node_imgs[b, k] = item["node_images"][v_name]
            node_img_mask[b, k] = True

        if "edge_labels" in item:
            has_labels = True
            edge_labels[b, :E] = item["edge_labels"]
            K = len(other)
            node_labels[b, :K] = item["node_labels"]

    out = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "edge_mask": edge_mask,
        "struct_rel": struct_rel,
        "cols": cols_list,
        "node_imgs": node_imgs,
        "node_img_mask": node_img_mask,
    }
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels
    return out


# ============================================================
# Model Architecture (identical to v26e)
# ============================================================
class ConvBlock1D(nn.Module):
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
    def __init__(self, d, n_channels):
        super().__init__()
        self.linear = nn.Linear(n_channels, d)

    def forward(self, x):
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class EdgeFeatureExtractor(nn.Module):
    def __init__(self, d=64, n_blocks=5, n_channels=8):
        super().__init__()
        self.stem = StemLayer(d, n_channels)
        self.blocks = nn.Sequential(*[ConvBlock1D(d) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


class NodeFeatureExtractor2D(nn.Module):
    def __init__(self, d=64, n_channels=12):
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
    def __init__(self, d=None, n_edge_types=None, aug_noise_std=0.0):
        super().__init__()
        d = d or D_MODEL
        n_edge_types = n_edge_types or N_EDGE_TYPES
        self.d = d
        self.aug_noise_std = aug_noise_std

        self.extractor = EdgeFeatureExtractor(d, n_channels=N_CHANNELS_1D)
        self.edge_type_emb = nn.Embedding(n_edge_types, d)
        self.edge_merge = MergeOperator(n_inputs=2, d=d)
        self.attn_layers = nn.ModuleList(
            [StructuralSelfAttention(d, n_heads=N_HEADS, n_struct_types=N_STRUCT_TYPES)
             for _ in range(2)]
        )
        self.edge_head = nn.Linear(d, 2)

        self.node_extractor = NodeFeatureExtractor2D(d, n_channels=N_NODE_CHANNELS)
        self.node_merge = MergeOperator(n_inputs=5, d=d)
        self.node_head = nn.Linear(d, N_CLASSES)

    def forward(self, edge_data, edge_types, edge_mask, cols_list,
                node_imgs=None, node_img_mask=None, struct_rel=None):
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

        node_logits_list = []
        if node_imgs is not None:
            B_n, K, C_n, H, W = node_imgs.shape
            flat_imgs = node_imgs.view(B_n * K, C_n, H, W)
            flat_emb = self.node_extractor(flat_imgs)
            node_2d_embs = flat_emb.view(B_n, K, self.d)

            def _eidx(p, u, v):
                idx = 0
                for uu in range(p):
                    for vv in range(p):
                        if uu == vv:
                            continue
                        if uu == u and vv == v:
                            return idx
                        idx += 1
                return -1

            for b in range(B):
                cols = cols_list[b]
                p = len(cols)
                col2idx = {c: i for i, c in enumerate(cols)}
                xi = col2idx["X"]
                yi = col2idx["Y"]
                other_nodes = [c for c in cols if c not in ("X", "Y")]
                K_b = int(node_img_mask[b].sum().item()) if node_img_mask is not None else len(other_nodes)

                if K_b > 0:
                    per_node_logits = []
                    for k, v_name in enumerate(other_nodes[:K_b]):
                        vi = col2idx[v_name]
                        e_vx = _eidx(p, vi, xi)
                        e_vy = _eidx(p, vi, yi)
                        e_xv = _eidx(p, xi, vi)
                        e_yv = _eidx(p, yi, vi)

                        embs = [
                            edge_emb[b, e_vx],
                            edge_emb[b, e_vy],
                            edge_emb[b, e_xv],
                            edge_emb[b, e_yv],
                            node_2d_embs[b, k],
                        ]
                        merged = self.node_merge(embs)
                        per_node_logits.append(self.node_head(merged))

                    node_logits_list.append(torch.stack(per_node_logits, dim=0))
                else:
                    node_logits_list.append(None)
        else:
            node_logits_list = [None] * B

        return edge_logits, node_logits_list


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
    def __init__(self, d=None, node_class_weights=None, edge_class_weights=None,
                 lr=1e-3, max_epochs=20, aug_noise_std=0.0):
        super().__init__()
        d = d or D_MODEL
        self.model = ADIAModel(d, aug_noise_std=aug_noise_std)
        self.lr = lr
        self.max_epochs = max_epochs
        self.node_criterion = nn.CrossEntropyLoss(
            weight=node_class_weights if node_class_weights is not None else torch.ones(N_CLASSES),
            ignore_index=-1,
        )
        self.edge_criterion = nn.CrossEntropyLoss(
            weight=edge_class_weights if edge_class_weights is not None else torch.ones(2),
            ignore_index=-1,
        )

    def forward(self, batch):
        return self.model(
            batch["edge_data"].to(self.device),
            batch["edge_types"].to(self.device),
            batch["edge_mask"].to(self.device),
            batch["cols"],
            node_imgs=batch["node_imgs"].to(self.device),
            node_img_mask=batch["node_img_mask"].to(self.device),
            struct_rel=batch["struct_rel"].to(self.device),
        )

    def _compute_loss(self, batch, split):
        edge_logits, node_logits_list = self(batch)
        B = edge_logits.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)

        # Edge loss (lambda-weighted auxiliary)
        if "edge_labels" in batch:
            el = batch["edge_labels"].to(self.device)
            mask = batch["edge_mask"].to(self.device)
            fl, fb = edge_logits[mask], el[mask]
            if fl.numel() > 0:
                edge_loss = self.edge_criterion(fl, fb)
                total_loss = total_loss + LOSS_LAMBDA * edge_loss
                self.log(f"{split}/edge_loss", edge_loss, prog_bar=False, sync_dist=True)

        # Node loss (primary)
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
                    node_loss = self.node_criterion(cat_l[valid], cat_n[valid])
                    total_loss = total_loss + (1 - LOSS_LAMBDA) * node_loss
                    self.log(f"{split}/node_loss", node_loss, prog_bar=True, sync_dist=True)

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
    edge_w_path = os.path.join(LOCAL_CACHE_DIR, "edge_weights_v8b.pt")
    if os.path.exists(node_w_path) and os.path.exists(edge_w_path):
        node_w = torch.load(node_w_path, weights_only=True)
        edge_w = torch.load(edge_w_path, weights_only=True)
    else:
        node_w = compute_class_weights(y_list)
        edge_w = compute_edge_class_weights(y_list)
        torch.save(node_w, node_w_path)
        torch.save(edge_w, edge_w_path)

    shard_dir = os.path.join(LOCAL_CACHE_DIR, f"train_dataset_{VERSION_NAME}")
    if not os.path.exists(os.path.join(shard_dir, "meta.pkl")):
        _build_and_save_shards(X_list, y_list, shard_dir, n_workers=47)

    dataset = CausalEdgeDataset(shard_dir)
    sampler = ShardGroupedSampler(dataset.shard_sizes)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=0, collate_fn=collate_fn, pin_memory=True)

    print(f"Dataset: {len(dataset)} samples, {dataset.n_shards} shards")

    wrapper = ADIAModelWrapper(d=D_MODEL, node_class_weights=node_w, edge_class_weights=edge_w,
                                lr=LR, max_epochs=MAX_EPOCHS, aug_noise_std=AUG_NOISE_STD)
    n_params = sum(p.numel() for p in wrapper.model.parameters())
    print(f"Params: {n_params:,}")

    wandb_logger = WandbLogger(
        project="causal-discovery-thesis",
        name=VERSION_NAME,
        config={
            "version": VERSION_NAME,
            "n_node_channels": N_NODE_CHANNELS,
            "scatter_sigma": SCATTER_SIGMA,
            "anm_sigma": ANM_SIGMA,
            "anm_bw": ANM_BW,
            "loss_lambda": LOSS_LAMBDA,
            "d_model": D_MODEL,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_epochs": MAX_EPOCHS,
            "n_params": n_params,
            "n_samples": len(dataset),
            "xy_aug": True,
        },
    )

    trainer = pl.Trainer(
        accelerator="gpu", devices=2,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=MAX_EPOCHS, precision="32-true",
        use_distributed_sampler=False,
        logger=wandb_logger, enable_checkpointing=True, enable_progress_bar=True,
    )
    trainer.fit(wrapper, loader)

    path = os.path.join(model_directory_path, f"{MODEL_NAME}.pt")
    sd = wrapper.model.state_dict()
    torch.save({k.replace("module.", ""): v for k, v in sd.items()}, path)
    print(f"Model saved to {path}")


@torch.no_grad()
def infer_batch_local(dfs, model, device="cpu", batch_size=32, cache_dir=None):
    model = model.eval()
    cache_path = os.path.join(cache_dir, f"infer_{VERSION_NAME}.pkl") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            all_items = pickle.load(f)
    else:
        import multiprocessing as mp
        n_workers = max(1, mp.cpu_count() - 1)
        args = [(df, None) for df in dfs]
        print(f"Building edge+node tensors infer ({len(dfs)} graphs, {n_workers} workers)...")
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

    from torch.utils.data import DataLoader as DL

    class _InferDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            item = self.samples[idx]
            return {
                "edge_data": torch.from_numpy(item["edge_data"]),
                "edge_types": torch.from_numpy(item["edge_types"]),
                "cols": item["cols"],
                "p": item["p"],
                "other_nodes": item["other_nodes"],
                "node_images": {k: torch.from_numpy(v) for k, v in item["node_images"].items()},
            }

    loader = DL(_InferDataset(all_items), batch_size=batch_size,
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
            node_imgs=batch["node_imgs"].to(device),
            node_img_mask=batch["node_img_mask"].to(device),
            struct_rel=batch["struct_rel"].to(device))
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


def infer(X_test, model_directory_path, id_column_name, prediction_column_name):
    path = os.path.join(model_directory_path, f"{MODEL_NAME}.pt")
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
    print(f"{VERSION}: 12ch Density + Edge Context + XY Augmentation")
    print(f"  Node: {N_NODE_CHANNELS}ch x {GRID_SIZE}x{GRID_SIZE}")
    print(f"  scatter_sigma={SCATTER_SIGMA}, anm_sigma={ANM_SIGMA}")
    print(f"  loss_lambda={LOSS_LAMBDA}")
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
    model.load_state_dict(torch.load(f"resources/{MODEL_NAME}.pt", map_location=device, weights_only=True))
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
    print(f"\n{MODEL_NAME} Per-class accuracy:")
    accs = []
    for cls in CLASS_NAMES:
        n = ct[cls]
        acc = cc[cls] / n if n > 0 else 0.0
        accs.append(acc)
        print(f"  {cls:25s}: {acc:.4f}  (n={n})")
    print(f"\n{MODEL_NAME} Balanced Accuracy: {np.mean(accs):.4f}")
