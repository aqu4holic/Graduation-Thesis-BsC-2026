"""
v5_augstat.py — ADIA Causal Discovery
  v2 baseline + edge stat injection + X/Y remap augmentation + shard-based training

Key features:
  - Edge-level statistics (partial corr, residual asymmetry) injected after conv pooling
  - X/Y remap augmentation (~11x data) using all directed edges in ground truth DAG
  - Sharded dataset: only one shard (~50K samples) in RAM at a time
  - ShardGroupedSampler: groups indices by shard so Dataset never thrashes
  - Standard PL training — works with DDP multi-GPU out of the box
"""

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

N_EDGE_STATS = 2
AUG_NOISE_STD = 0.01
N_AUG = 1
MAX_EPOCHS = 5
BATCH_SIZE = 64
LR = 2e-3
LOCAL_CACHE_DIR = "dataset_cache/"
SHARD_SIZE = 50_000


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
        submatrix = adjacency_matrix.loc[[variable, "X", "Y"], [variable, "X", "Y"]]
        key = tuple(submatrix.values.flatten())
        result[variable] = adjacency_label[key]
    return result

def transform_proba_to_DAG(nodes, pred):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edge("X", "Y")
    x_index, y_index = np.unravel_index(np.argsort(pred.ravel())[::-1], pred.shape)
    for i, j in zip(x_index, y_index):
        n1, n2 = nodes[i], nodes[j]
        if i == j: continue
        if {n1, n2} == {"X", "Y"}: continue
        if pred[i, j] > 0.5:
            G.add_edge(n1, n2)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(n1, n2)
    return nx.to_numpy_array(G)


# ============================================================
# Data Preprocessing
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

def compute_multivariate_kernel_coefficients(data, n_sub=None, bandwidth=0.5):
    if n_sub is None: n_sub = N_KERNEL
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

def compute_edge_statistics(data):
    N, p = data.shape
    cov = np.cov(data.T) + 1e-6 * np.eye(p)
    try: precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError: precision = np.eye(p)
    diag = np.sqrt(np.maximum(np.diag(precision), 1e-10))
    pcorr = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcorr, 0.0)
    resid_corr = np.zeros((p, p), dtype=np.float32)
    for i in range(p):
        xi = data[:, i]; xi_var = np.var(xi)
        if xi_var < 1e-10: continue
        xi_mean = xi.mean()
        for j in range(p):
            if i == j: continue
            xj = data[:, j]
            b = np.cov(xi, xj)[0, 1] / (xi_var + 1e-10)
            a = xj.mean() - b * xi_mean
            residuals = xj - (a + b * xi)
            r_std = np.std(residuals)
            resid_corr[i, j] = 0.0 if r_std < 1e-10 else np.corrcoef(residuals, xi)[0, 1]
    return pcorr.astype(np.float32), resid_corr

def build_edge_tensor(df):
    cols = list(df.columns); p = len(cols)
    data = df.values.astype(np.float32); N = data.shape[0]
    coeff_map = compute_multivariate_kernel_coefficients(data)
    pcorr_matrix, resid_matrix = compute_edge_statistics(data)
    edges, edge_types, edge_stats = [], [], []
    for i, u_name in enumerate(cols):
        sort_idx = np.argsort(data[:, i]); u_sorted = data[sort_idx, i]
        for j, v_name in enumerate(cols):
            if i == j: continue
            v_sorted_by_u = data[sort_idx, j]
            coeff_sorted = coeff_map[(i, j)][sort_idx]
            edges.append(np.stack([u_sorted, v_sorted_by_u, coeff_sorted], axis=0))
            edge_types.append(_edge_type(u_name, v_name))
            edge_stats.append(np.array([pcorr_matrix[i, j], resid_matrix[i, j]], dtype=np.float32))
    return (np.stack(edges, axis=0).astype(np.float32),
            np.array(edge_types, dtype=np.int64),
            np.stack(edge_stats, axis=0).astype(np.float32))

def _build_single(args):
    df, y_df = args
    edge_data, edge_types, edge_stats = build_edge_tensor(df)
    cols = list(df.columns); p = len(cols)
    result = {"edge_data": edge_data, "edge_types": edge_types,
              "edge_stats": edge_stats, "cols": cols, "p": p}
    if y_df is not None:
        adj_np = y_df.values.astype(np.float32); adj_cols = list(y_df.columns)
        result["adj"] = adj_np
        edge_labels = []
        for ui in range(p):
            for vi in range(p):
                if ui != vi: edge_labels.append(int(adj_np[ui, vi]))
        result["edge_labels"] = np.array(edge_labels, dtype=np.int64)
        _, adjacency_label = create_graph_label()
        df_adj = pd.DataFrame(adj_np, columns=adj_cols, index=adj_cols)
        other_nodes = [v for v in adj_cols if v not in ("X", "Y")]
        node_labels = []
        for v in other_nodes:
            sub = df_adj.loc[[v, "X", "Y"], [v, "X", "Y"]]
            key = tuple(sub.values.flatten())
            node_labels.append(CLASS_NAMES.index(adjacency_label.get(key, "Independent")))
        result["node_labels"] = np.array(node_labels, dtype=np.int64)
        result["other_nodes"] = other_nodes
    return result

def _remap_xy_names(cols, new_x, new_y):
    result = list(cols); pos = {c: i for i, c in enumerate(cols)}
    i_nx, i_x = pos[new_x], pos["X"]
    result[i_nx], result[i_x] = result[i_x], result[i_nx]
    pos2 = {c: i for i, c in enumerate(result)}
    i_ny, i_y = pos2[new_y], pos2["Y"]
    result[i_ny], result[i_y] = result[i_y], result[i_ny]
    return {cols[i]: result[i] for i in range(len(cols))}

def _build_augmented(args):
    df, y_df, _ = args
    results = [_build_single((df, y_df))]
    if y_df is None: return results
    cols = list(df.columns); adj_np = y_df.values
    col_idx = {c: i for i, c in enumerate(y_df.columns)}
    for a_name in y_df.columns:
        for b_name in y_df.columns:
            if a_name == b_name: continue
            if adj_np[col_idx[a_name], col_idx[b_name]] != 1: continue
            if a_name == "X" and b_name == "Y": continue
            rename_map = _remap_xy_names(cols, a_name, b_name)
            results.append(_build_single((df.rename(columns=rename_map),
                                          y_df.rename(index=rename_map, columns=rename_map))))
    return results


# ============================================================
# Sharded Dataset + Sampler
# ============================================================
def _save_sharded(samples, shard_dir):
    os.makedirs(shard_dir, exist_ok=True)
    n_shards = (len(samples) + SHARD_SIZE - 1) // SHARD_SIZE
    shard_sizes = []
    for i in range(n_shards):
        start, end = i * SHARD_SIZE, min((i+1) * SHARD_SIZE, len(samples))
        with open(os.path.join(shard_dir, f"shard_{i:04d}.pkl"), "wb") as f:
            pickle.dump(samples[start:end], f, protocol=pickle.HIGHEST_PROTOCOL)
        shard_sizes.append(end - start)
        print(f"  Saved shard {i+1}/{n_shards}: {end-start} samples")
    meta = {"n_samples": len(samples), "n_shards": n_shards, "shard_sizes": shard_sizes}
    with open(os.path.join(shard_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved {len(samples)} samples in {n_shards} shards.")
    return meta

def _build_and_save_shards(X_list, y_list, shard_dir, n_workers=10, augment=False):
    use_aug = augment and y_list is not None
    if use_aug:
        args = [(X_list[i], y_list[i], 0) for i in range(len(X_list))]
        print(f"Building augmented dataset ({len(args)} base, X/Y remap, {n_workers} workers)...")
        ctx = __import__('multiprocessing').get_context('fork')
        if n_workers > 1:
            raw_nested = [None] * len(args)
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {pool.submit(_build_augmented, a): idx for idx, a in enumerate(args)}
                for fut in tqdm(as_completed(futures), total=len(args)):
                    raw_nested[futures[fut]] = fut.result()
        else:
            raw_nested = [_build_augmented(a) for a in tqdm(args)]
        raw = []
        for group in raw_nested:
            raw.extend(group)
        print(f"Total samples: {len(raw)} ({len(raw)/len(args):.1f}x avg)")
    else:
        args = [(X_list[i], y_list[i] if y_list else None) for i in range(len(X_list))]
        print(f"Building dataset ({len(args)} samples, {n_workers} workers)...")
        ctx = __import__('multiprocessing').get_context('fork')
        if n_workers > 1:
            raw = [None] * len(args)
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {pool.submit(_build_single, a): idx for idx, a in enumerate(args)}
                for fut in tqdm(as_completed(futures), total=len(args)):
                    raw[futures[fut]] = fut.result()
        else:
            raw = [_build_single(a) for a in tqdm(args)]
    meta = _save_sharded(raw, shard_dir)
    del raw; gc.collect()
    return meta


class ShardGroupedSampler(Sampler):
    """Yields indices grouped by shard: all of shard 0, then shard 1, etc.
    Shuffles shard order + within-shard order each epoch."""
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
    """Shard-based dataset. Only one shard loaded at a time.
    Paired with ShardGroupedSampler for sequential shard access."""
    def __init__(self, shard_dir):
        with open(os.path.join(shard_dir, "meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)
        self.shard_dir = shard_dir
        self.n_samples = self.meta["n_samples"]
        self.n_shards = self.meta["n_shards"]
        self.shard_sizes = self.meta["shard_sizes"]
        self.shard_offsets = []
        offset = 0
        for s in self.shard_sizes:
            self.shard_offsets.append(offset)
            offset += s
        self._loaded_shard_idx = -1
        self._loaded_data = None

    def _load_shard(self, shard_idx):
        if shard_idx == self._loaded_shard_idx:
            return
        self._loaded_data = None
        gc.collect()
        path = os.path.join(self.shard_dir, f"shard_{shard_idx:04d}.pkl")
        with open(path, "rb") as f:
            self._loaded_data = pickle.load(f)
        self._loaded_shard_idx = shard_idx
        print(f"  [Dataset] Loaded shard {shard_idx} ({len(self._loaded_data)} samples)")

    def _find_shard(self, global_idx):
        for i in range(self.n_shards):
            if global_idx < self.shard_offsets[i] + self.shard_sizes[i]:
                return i
        return self.n_shards - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        shard_idx = self._find_shard(idx)
        self._load_shard(shard_idx)
        local_idx = idx - self.shard_offsets[shard_idx]
        item = self._loaded_data[local_idx]
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
        return sample


class InMemoryDataset(Dataset):
    """Simple in-memory dataset for inference (small data)."""
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
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
        return sample


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
    cols_list = []; has_labels = False
    for b, item in enumerate(batch):
        E = item["edge_data"].shape[0]
        edge_data[b, :E] = item["edge_data"]; edge_types[b, :E] = item["edge_types"]
        edge_mask[b, :E] = True; edge_stats[b, :E] = item["edge_stats"]
        cols_list.append(item["cols"])
        if "edge_labels" in item:
            has_labels = True; edge_labels[b, :E] = item["edge_labels"]
            K = item["node_labels"].shape[0]
            node_labels[b, :K] = item["node_labels"]; node_mask[b, :K] = True
    out = {"edge_data": edge_data, "edge_types": edge_types,
           "edge_mask": edge_mask, "edge_stats": edge_stats, "cols": cols_list}
    if has_labels:
        out["edge_labels"] = edge_labels; out["node_labels"] = node_labels; out["node_mask"] = node_mask
    return out


# ============================================================
# Model
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, d, kernel_size=3, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(n_groups, d); self.act = nn.GELU()
    def forward(self, x): return x + self.act(self.norm(self.conv(x)))

class MergeOperator(nn.Module):
    def __init__(self, n_inputs, d):
        super().__init__()
        self.linear = nn.Linear(n_inputs * d, d); self.norm = nn.LayerNorm(d); self.act = nn.GELU()
    def forward(self, inputs): return self.act(self.norm(self.linear(torch.cat(inputs, dim=-1))))

class StemLayer(nn.Module):
    def __init__(self, d):
        super().__init__(); self.linear = nn.Linear(3, d)
    def forward(self, x): return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

class EdgeFeatureExtractor(nn.Module):
    def __init__(self, d=64, n_blocks=5):
        super().__init__(); self.stem = StemLayer(d)
        self.blocks = nn.Sequential(*[ConvBlock(d) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x); return self.pool(x).squeeze(-1)

class StatProjector(nn.Module):
    def __init__(self, n_stats, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_stats, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
    def forward(self, stats): return self.proj(stats)

class SelfAttentionLayer(nn.Module):
    def __init__(self, d=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.norm2 = nn.LayerNorm(d)
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out); x = self.norm2(x + self.ff(x)); return x

class ADIAModel(nn.Module):
    def __init__(self, d=None, n_edge_types=None, n_edge_stats=None, aug_noise_std=0.0):
        super().__init__()
        d = d or D_MODEL; n_edge_types = n_edge_types or N_EDGE_TYPES
        n_edge_stats = n_edge_stats or N_EDGE_STATS
        self.d = d; self.aug_noise_std = aug_noise_std
        self.extractor = EdgeFeatureExtractor(d)
        self.stat_proj = StatProjector(n_edge_stats, d)
        self.edge_type_emb = nn.Embedding(n_edge_types, d)
        self.edge_merge = MergeOperator(n_inputs=3, d=d)
        self.attn_layers = nn.ModuleList([SelfAttentionLayer(d) for _ in range(2)])
        self.edge_head = nn.Linear(d, 2)
        self.node_merge = MergeOperator(n_inputs=4, d=d)
        self.node_head = nn.Linear(d, N_CLASSES)

    def forward(self, edge_data, edge_types, edge_mask, cols_list, edge_stats=None):
        B, E, C, N = edge_data.shape
        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std
        edge_emb = self.extractor(edge_data.view(B*E, C, N)).view(B, E, self.d)
        type_emb = self.edge_type_emb(edge_types)
        stat_emb = self.stat_proj(edge_stats) if edge_stats is not None else torch.zeros_like(edge_emb)
        edge_emb = self.edge_merge([edge_emb, type_emb, stat_emb])
        pad_mask = ~edge_mask
        for attn in self.attn_layers: edge_emb = attn(edge_emb, key_padding_mask=pad_mask)
        edge_logits = self.edge_head(edge_emb)
        node_logits_list = []
        for b in range(B):
            cols = cols_list[b]; p = len(cols)
            col_idx = {name: i for i, name in enumerate(cols)}
            edge_order = {}; count = 0
            for ui in range(p):
                for vi in range(p):
                    if ui != vi: edge_order[(ui, vi)] = count; count += 1
            x_idx, y_idx = col_idx.get("X"), col_idx.get("Y")
            embs = edge_emb[b]; other_nodes = [n for n in cols if n not in ("X", "Y")]
            if not other_nodes or x_idx is None or y_idx is None:
                node_logits_list.append(None); continue
            node_logits = []
            for node_name in other_nodes:
                u = col_idx[node_name]
                gathered = [embs[edge_order[(u, x_idx)]], embs[edge_order[(u, y_idx)]],
                           embs[edge_order[(x_idx, u)]], embs[edge_order[(y_idx, u)]]]
                node_logits.append(self.node_head(self.node_merge(gathered)))
            node_logits_list.append(torch.stack(node_logits) if node_logits else None)
        return edge_logits, node_logits_list


# ============================================================
# Training Wrapper
# ============================================================
def compute_class_weights(y_list):
    adjacency_label = get_adjacency_label(); counts = torch.zeros(N_CLASSES)
    for y_df in y_list:
        cols = list(y_df.columns); arr = y_df.values
        col_idx = {c: i for i, c in enumerate(cols)}
        for v in cols:
            if v in ("X", "Y"): continue
            idx = [col_idx[v], col_idx["X"], col_idx["Y"]]
            sub = arr[np.ix_(idx, idx)]; key = tuple(sub.flatten())
            counts[CLASS_NAMES.index(adjacency_label.get(key, "Independent"))] += 1
    w = 1.0 / (counts + 1e-6); return w / w.sum() * N_CLASSES

def compute_edge_class_weights(y_list):
    counts = torch.zeros(2)
    for y_df in y_list:
        arr = y_df.values; p = arr.shape[0]
        for i in range(p):
            for j in range(p):
                if i != j: counts[int(arr[i, j])] += 1
    w = 1.0 / (counts + 1e-6); return w / w.sum() * 2

class ADIAModelWrapper(pl.LightningModule):
    def __init__(self, d=None, node_class_weights=None, edge_class_weights=None,
                 lr=2e-3, max_epochs=20, aug_noise_std=0.0):
        super().__init__()
        d = d or D_MODEL
        self.model = ADIAModel(d, aug_noise_std=aug_noise_std)
        self.lr = lr; self.max_epochs = max_epochs
        self.node_criterion = nn.CrossEntropyLoss(
            weight=node_class_weights if node_class_weights is not None
            else torch.ones(N_CLASSES), ignore_index=-1)
        self.edge_criterion = nn.CrossEntropyLoss(
            weight=edge_class_weights if edge_class_weights is not None
            else torch.ones(2), ignore_index=-1)

    def forward(self, batch):
        return self.model(batch["edge_data"].to(self.device), batch["edge_types"].to(self.device),
                          batch["edge_mask"].to(self.device), batch["cols"],
                          edge_stats=batch["edge_stats"].to(self.device))

    def _compute_loss(self, batch, split):
        edge_logits, node_logits_list = self(batch); B = edge_logits.shape[0]
        total_loss = torch.tensor(0.0, device=self.device); n_terms = 0
        if "edge_labels" in batch:
            el = batch["edge_labels"].to(self.device)
            total_loss = total_loss + self.edge_criterion(edge_logits.view(-1, 2), el.view(-1)); n_terms += 1
        if "node_labels" in batch:
            nl = batch["node_labels"].to(self.device)
            all_logits, all_labels = [], []
            for b in range(B):
                if node_logits_list[b] is not None:
                    K = node_logits_list[b].shape[0]
                    all_logits.append(node_logits_list[b]); all_labels.append(nl[b, :K])
            if all_logits:
                total_loss = total_loss + self.node_criterion(
                    torch.cat(all_logits, 0), torch.cat(all_labels, 0)); n_terms += 1
        if n_terms > 0: total_loss = total_loss / n_terms
        self.log(f"{split}_loss", total_loss, on_step=(split=="train"),
                 on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx): return self._compute_loss(batch, "train")
    def validation_step(self, batch, batch_idx): return self._compute_loss(batch, "val")

    def on_train_epoch_start(self):
      sampler = self.trainer.train_dataloader.sampler
      if hasattr(sampler, 'set_epoch'):
          sampler.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def infer_batch_local(dfs, model, device="cpu", batch_size=32, cache_dir=None):
    model = model.eval()
    cache_path = os.path.join(cache_dir, f"infer_edge_tensors_v5_nk{N_KERNEL}.pkl") if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached edge tensors from {cache_path}...")
        with open(cache_path, "rb") as f: all_items = pickle.load(f)
    else:
        import multiprocessing as mp; n_workers = max(1, mp.cpu_count() - 1)
        args = [(df, None) for df in dfs]
        print(f"Building edge tensors ({len(dfs)} graphs, {n_workers} workers)...")
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            all_items = list(tqdm(pool.map(_build_single, args, chunksize=8), total=len(args)))
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f: pickle.dump(all_items, f, protocol=pickle.HIGHEST_PROTOCOL)

    infer_ds = InMemoryDataset(all_items)
    loader = DataLoader(infer_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    results = [None] * len(all_items)
    patterns = {
        "Confounder": lambda n: [(n,"X"),(n,"Y")], "Collider": lambda n: [("X",n),("Y",n)],
        "Mediator": lambda n: [("X",n),(n,"Y")], "Cause of X": lambda n: [(n,"X")],
        "Cause of Y": lambda n: [(n,"Y")], "Consequence of X": lambda n: [("X",n)],
        "Consequence of Y": lambda n: [("Y",n)], "Independent": lambda n: [],
    }
    idx_offset = 0
    for batch in tqdm(loader, desc="infering"):
        edge_logits, node_logits_list = model(
            batch["edge_data"].to(device), batch["edge_types"].to(device),
            batch["edge_mask"].to(device), batch["cols"],
            edge_stats=batch["edge_stats"].to(device))
        for b in range(edge_logits.shape[0]):
            item = all_items[idx_offset + b]; cols, p = item["cols"], item["p"]
            edge_probs = torch.softmax(edge_logits[b], dim=-1)[:, 1]
            E_mat = np.zeros((p, p)); count = 0
            for ii in range(p):
                for jj in range(p):
                    if ii != jj: E_mat[ii, jj] = edge_probs[count].item(); count += 1
            adj = transform_proba_to_DAG(cols, E_mat).astype(int)
            A = pd.DataFrame(adj, columns=cols, index=cols)
            if node_logits_list[b] is not None:
                other_nodes = [n for n in cols if n not in ("X", "Y")]
                node_preds = torch.argmax(node_logits_list[b], dim=-1)
                for k, node_name in enumerate(other_nodes):
                    pred_class = CLASS_NAMES[node_preds[k].item()]
                    A.loc[node_name, :] = 0; A.loc[:, node_name] = 0
                    for (src, dst) in patterns[pred_class](node_name): A.loc[src, dst] = 1
            results[idx_offset + b] = A
        idx_offset += edge_logits.shape[0]
    return results


# ============================================================
# Train & Infer
# ============================================================
def train(X_train, y_train, model_directory_path):
    keys = list(X_train.keys())
    X_list = [X_train[k] for k in keys]; y_list = [y_train[k] for k in keys]
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    node_w_path = os.path.join(LOCAL_CACHE_DIR, "node_weights.pt")
    edge_w_path = os.path.join(LOCAL_CACHE_DIR, "edge_weights.pt")
    if os.path.exists(node_w_path) and os.path.exists(edge_w_path):
        node_w = torch.load(node_w_path, weights_only=True)
        edge_w = torch.load(edge_w_path, weights_only=True)
    else:
        node_w = compute_class_weights(y_list); edge_w = compute_edge_class_weights(y_list)
        torch.save(node_w, node_w_path); torch.save(edge_w, edge_w_path)

    aug_tag = "xyaug" if N_AUG > 0 else "noaug"
    shard_dir = os.path.join(LOCAL_CACHE_DIR, f"train_dataset_v5_{aug_tag}_nk{N_KERNEL}_shards")
    if not os.path.exists(os.path.join(shard_dir, "meta.pkl")):
        _build_and_save_shards(X_list, y_list, shard_dir, n_workers=47, augment=(N_AUG > 0))

    dataset = CausalEdgeDataset(shard_dir)
    sampler = ShardGroupedSampler(dataset.shard_sizes)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)

    print(f"Dataset: {len(dataset)} samples, {dataset.n_shards} shards")

    wrapper = ADIAModelWrapper(d=D_MODEL, node_class_weights=node_w, edge_class_weights=edge_w,
                               lr=LR, max_epochs=MAX_EPOCHS, aug_noise_std=AUG_NOISE_STD)
    print(f"Params: {sum(p.numel() for p in wrapper.model.parameters()):,}")

    trainer = pl.Trainer(
        accelerator="gpu", devices=2, strategy="ddp",
        max_epochs=MAX_EPOCHS, precision="32-true", use_distributed_sampler=False,
        logger=True, enable_checkpointing=True, enable_progress_bar=True)

    trainer.fit(wrapper, loader, ckpt_path="lightning_logs/version_8/checkpoints/epoch=2-step=12345.ckpt")

    path = os.path.join(model_directory_path, "model.pt")
    sd = wrapper.model.state_dict()
    torch.save({k.replace("module.", ""): v for k, v in sd.items()}, path)
    print(f"Model saved to {path}")


def infer(X_test, model_directory_path, id_column_name, prediction_column_name):
    path = os.path.join(model_directory_path, "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL, aug_noise_std=0.0)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    dfs = [X_test[n] for n in X_test]
    adj_list = infer_batch_local(dfs, model, device=device, batch_size=64, cache_dir=LOCAL_CACHE_DIR)
    submission = {}
    for name, A in zip(X_test.keys(), adj_list):
        for i in A.columns:
            for j in A.columns:
                submission[f"{name}_{i}_{j}"] = int(A.loc[i, j])
    s = pd.Series(submission).reset_index(); s.columns = [id_column_name, prediction_column_name]
    return s


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    X_train = pd.read_pickle("data/X_train.pickle")
    y_train = pd.read_pickle("data/y_train.pickle")
    print(f"Loaded {len(X_train)} training samples.")
    train(X_train, y_train, model_directory_path="resources")

    X_test = pd.read_pickle("data/X_test_reduced.pickle")
    y_test = pd.read_pickle("data/y_test.pickle")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ADIAModel(d=D_MODEL, aug_noise_std=0.0)
    model.load_state_dict(torch.load("resources/model.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    dfs = [X_test[n] for n in X_test]; names = list(X_test.keys())
    adj_list = infer_batch_local(dfs, model, device=device, batch_size=64, cache_dir=LOCAL_CACHE_DIR)
    adjacency_label = get_adjacency_label()
    class_correct = {c: 0 for c in CLASS_NAMES}; class_total = {c: 0 for c in CLASS_NAMES}
    for name, A_pred in zip(names, adj_list):
        y_df = y_test[name]
        pred_labels = get_labels(A_pred, adjacency_label)
        true_labels = get_labels(y_df, adjacency_label)
        for v in true_labels:
            class_total[true_labels[v]] += 1
            if pred_labels.get(v, "Independent") == true_labels[v]: class_correct[true_labels[v]] += 1
    print("\nPer-class accuracy:"); accs = []
    for cls in CLASS_NAMES:
        n = class_total[cls]; acc = class_correct[cls] / n if n > 0 else 0.0
        accs.append(acc); print(f"  {cls:25s}: {acc:.4f}  (n={n})")
    print(f"\nBalanced Accuracy: {np.mean(accs):.4f}")
