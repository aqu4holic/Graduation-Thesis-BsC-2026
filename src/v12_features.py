"""
v12_features.py — Graph-level feature extraction for the Graph Transformer.

For each observational graph, computes:
  ┌─ Per directed edge (i→j) ─────────────────────────────────────────────────┐
  │  sorted_u, v_sorted_by_u, kernel_coeff @ bw=0.2/0.5/1.0, ANM resid @ 3bw │
  │  → [n, n, 8, N_SUB] stored sequences  (8ch, subsampled to N_SUB=256)      │
  ├─ Per variable v ────────────────────────────────────────────────────────────┤
  │  variance, skewness, kurtosis  → node_stats [n, 3]                         │
  ├─ All-pairs MI ──────────────────────────────────────────────────────────────┤
  │  I(i; j)  → pairwise_mi [n, n]                                             │
  │  I(i; j | k) for all distinct triples  → cond_mi [n, n, n]                │
  ├─ PC algorithm (skeleton + CPDAG orientation) ───────────────────────────────┤
  │  pc_skel [n, n],  pc_dir [n, n]                                            │
  └─ DirectLiNGAM adjacency matrix ────────────────────────────────────────────┘
     lingam_B [n, n]

Output: list of GraphData objects written to shard pickle files.
"""

from __future__ import annotations

import os
import gc
import math
import pickle
import logging
import warnings
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import digamma
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis
from tqdm.auto import tqdm

# Must import before numpy forks to prevent BLAS deadlock
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
N_SUB        = 256          # subsample edge sequences to this length for storage
N_KERNEL     = 1000         # kernel regression subsample (same as v8b)
BANDWIDTHS   = [0.2, 0.5, 1.0]
CMI_K        = 5            # k-NN for MI/CMI estimator
CLASS_NAMES  = [
    "Confounder", "Collider", "Mediator",
    "Cause of X", "Cause of Y",
    "Consequence of X", "Consequence of Y",
    "Independent",
]


# ─── Causal label lookup (same as existing codebase) ──────────────────────────

def _create_adjacency_label() -> dict:
    graphs = {
        nx.DiGraph([("X", "Y"), ("v", "X"), ("v", "Y")]): "Confounder",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("Y", "v")]): "Collider",
        nx.DiGraph([("X", "Y"), ("X", "v"), ("v", "Y")]): "Mediator",
        nx.DiGraph([("X", "Y"), ("v", "X")]): "Cause of X",
        nx.DiGraph([("X", "Y"), ("v", "Y")]): "Cause of Y",
        nx.DiGraph([("X", "Y"), ("X", "v")]): "Consequence of X",
        nx.DiGraph([("X", "Y"), ("Y", "v")]): "Consequence of Y",
        nx.DiGraph([("X", "Y")]): "Independent",
    }
    adjacency_label: dict = {}
    for g, label in graphs.items():
        nodelist = ["v", "X", "Y"]
        mat = nx.adjacency_matrix(g, nodelist=nodelist).todense()
        key = tuple(np.array(mat).flatten())
        adjacency_label[key] = label
    return adjacency_label


_ADJACENCY_LABEL = _create_adjacency_label()


def get_node_label_for_v(adj_df: pd.DataFrame, v: str) -> int:
    """8-class label for variable v relative to X→Y in adj_df."""
    sub = adj_df.loc[[v, "X", "Y"], [v, "X", "Y"]]
    key = tuple(sub.values.flatten())
    label_str = _ADJACENCY_LABEL.get(key, "Independent")
    return CLASS_NAMES.index(label_str)


def get_all_node_labels(adj_df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """Return {v: class_idx} for all non-X, non-Y nodes."""
    result: dict[str, int] = {}
    for v in cols:
        if v in ("X", "Y"):
            continue
        result[v] = get_node_label_for_v(adj_df, v)
    return result


def get_xy_remap_assignments(adj_df: pd.DataFrame, cols: list[str]) -> list[tuple[str, str]]:
    """
    Return list of (x_name, y_name) for all directed edges in the ground truth DAG.
    Always includes the original ("X", "Y").
    """
    assignments = [("X", "Y")]
    adj_np = adj_df.values
    for i, src in enumerate(adj_df.columns):
        for j, tgt in enumerate(adj_df.columns):
            if i != j and adj_np[i, j] == 1 and (src, tgt) != ("X", "Y"):
                assignments.append((src, tgt))
    return assignments


def relabel_for_xy(adj_df: pd.DataFrame, cols: list[str], x_name: str, y_name: str) -> dict[str, int]:
    """
    Given new X=x_name, Y=y_name, recompute 8-class labels for all other nodes.
    Renames columns, then applies the same 4-bit lookup.
    """
    rename = {x_name: "X", y_name: "Y"}
    # We need a graph: just treat x_name→y_name as the X→Y edge
    # Rebuild a renamed adjacency
    renamed_df = adj_df.rename(index=rename, columns=rename)
    result: dict[str, int] = {}
    for v in cols:
        if v in (x_name, y_name):
            continue
        v_in_renamed = v
        try:
            sub = renamed_df.loc[[v_in_renamed, "X", "Y"], [v_in_renamed, "X", "Y"]]
            key = tuple(sub.values.flatten())
            label_str = _ADJACENCY_LABEL.get(key, "Independent")
            result[v] = CLASS_NAMES.index(label_str)
        except KeyError:
            result[v] = CLASS_NAMES.index("Independent")
    return result


# ─── k-NN MI / CMI (KSG estimator) ───────────────────────────────────────────

def _col(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 1) if x.ndim == 1 else x


def knn_mi(x: np.ndarray, y: np.ndarray, k: int = CMI_K) -> float:
    """I(X;Y) via KSG-1 estimator. Input arrays shape [N]."""
    x, y = _col(x.astype(np.float32)), _col(y.astype(np.float32))
    n = len(x)
    k = min(k, n - 2)
    if k < 1:
        return 0.0
    xy = np.hstack([x, y])
    eps = cKDTree(xy).query(xy, k=k + 1, workers=1, p=np.inf)[0][:, -1]
    nx_ = np.array([len(cKDTree(x).query_ball_point(x[i], max(eps[i] - 1e-15, 0), p=np.inf)) - 1
                    for i in range(n)], dtype=np.float32)
    ny_ = np.array([len(cKDTree(y).query_ball_point(y[i], max(eps[i] - 1e-15, 0), p=np.inf)) - 1
                    for i in range(n)], dtype=np.float32)
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx_ + 1) + digamma(ny_ + 1))
    return float(max(0.0, mi))


def knn_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = CMI_K) -> float:
    """I(X;Y|Z) via KSG conditional MI estimator."""
    x, y, z = (_col(v.astype(np.float32)) for v in (x, y, z))
    n = len(x)
    k = min(k, n - 2)
    if k < 1:
        return 0.0
    xz, yz, xyz = np.hstack([x, z]), np.hstack([y, z]), np.hstack([x, y, z])
    eps = cKDTree(xyz).query(xyz, k=k + 1, workers=1, p=np.inf)[0][:, -1]
    nxz = np.array([len(cKDTree(xz).query_ball_point(xz[i], max(eps[i]-1e-15, 0), p=np.inf)) - 1 for i in range(n)], dtype=np.float32)
    nyz = np.array([len(cKDTree(yz).query_ball_point(yz[i], max(eps[i]-1e-15, 0), p=np.inf)) - 1 for i in range(n)], dtype=np.float32)
    nz  = np.array([len(cKDTree(z ).query_ball_point(z [i], max(eps[i]-1e-15, 0), p=np.inf)) - 1 for i in range(n)], dtype=np.float32)
    cmi = digamma(k) + np.mean(digamma(nz + 1) - digamma(nxz + 1) - digamma(nyz + 1))
    return float(max(0.0, cmi))


def compute_pairwise_mi(data: np.ndarray, k: int = CMI_K) -> np.ndarray:
    """I(i;j) for all pairs. Returns [n, n] symmetric matrix."""
    n = data.shape[1]
    mi = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            v = knn_mi(data[:, i], data[:, j], k=k)
            mi[i, j] = mi[j, i] = v
    return mi


def compute_cond_mi_tensor(data: np.ndarray, k: int = CMI_K) -> np.ndarray:
    """I(i;j|k) for all distinct triples. Returns [n, n, n]."""
    n = data.shape[1]
    cmi = np.zeros((n, n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k_idx in range(n):
                if k_idx == i or k_idx == j:
                    continue
                cmi[i, j, k_idx] = knn_cmi(data[:, i], data[:, j], data[:, k_idx], k=k)
    return cmi


# ─── Kernel regression + ANM (from v8b) ───────────────────────────────────────

def compute_kernel_anm(
    data: np.ndarray,
    bandwidths: list[float] = BANDWIDTHS,
    n_sub: int = N_KERNEL,
) -> tuple[dict, dict]:
    """
    Returns (coeff_maps, resid_maps) each a list of dicts over bandwidths.
    coeff_maps[bw_idx][(k, j)] → (N,) kernel regression coeff for k→j
    resid_maps[bw_idx][j]      → (N,) residuals of j regressed on all others
    """
    N, p = data.shape
    n_sub = min(n_sub, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)
    data_sub = data[sub_idx]

    dist_to_sub = np.sum((data[:, None, :] - data_sub[None, :, :]) ** 2, axis=-1)  # (N, n_sub)
    nearest = np.argmin(dist_to_sub, axis=1)  # (N,)

    coeff_maps: list[dict] = []
    resid_maps: list[dict] = []

    for bw in bandwidths:
        sq_dist = np.sum((data_sub[:, None, :] - data_sub[None, :, :]) ** 2, axis=-1)
        W = np.exp(-sq_dist / (2 * bw ** 2))  # (n_sub, n_sub)

        all_coeffs: dict = {}
        for j in range(p):
            other_cols = [k for k in range(p) if k != j]
            X_d = np.concatenate([np.ones((n_sub, 1)), data_sub[:, other_cols]], axis=1)
            y_t = data_sub[:, j]
            A = np.einsum("il,la,lb->iab", W, X_d, X_d)
            b = np.einsum("il,la,l->ia",  W, X_d, y_t)
            reg = 1e-6 * np.eye(p)[None, :, :]
            try:
                c = np.linalg.solve(A + reg, b[:, :, None]).squeeze(-1)
            except np.linalg.LinAlgError:
                c = np.zeros((n_sub, p))
            all_coeffs[j] = (c, other_cols)

        cm: dict = {}
        rm: dict = {}
        for j in range(p):
            c, other_cols = all_coeffs[j]
            # Kernel coeff per directed pair
            for idx_in_other, k in enumerate(other_cols):
                cm[(k, j)] = c[:, idx_in_other + 1][nearest].astype(np.float32)
            # ANM residuals for target j
            c_nn = c[nearest]  # (N, p)
            X_full = np.concatenate([np.ones((N, 1)), data[:, other_cols]], axis=1)  # (N, p)
            y_hat = (c_nn * X_full).sum(axis=1)  # intercept + weighted sum
            rm[j] = (data[:, j] - y_hat).astype(np.float32)

        coeff_maps.append(cm)
        resid_maps.append(rm)

    return coeff_maps, resid_maps


def build_all_edge_sequences(
    data: np.ndarray,
    coeff_maps: list[dict],
    resid_maps: list[dict],
    n_sub_out: int = N_SUB,
) -> np.ndarray:
    """
    Build 8-channel sorted edge sequences for ALL directed pairs.

    Channels: sorted_u, v_sorted_by_u, coeff_bw0, coeff_bw1, coeff_bw2,
              anm_resid_bw0, anm_resid_bw1, anm_resid_bw2

    Returns: [n, n, 8, n_sub_out] float32 (zeros on diagonal)
    """
    N, p = data.shape
    out = np.zeros((p, p, 8, n_sub_out), dtype=np.float32)

    # Fixed subsample indices for consistent output length
    sub_idx = np.linspace(0, N - 1, n_sub_out, dtype=int)

    for i in range(p):
        sort_idx = np.argsort(data[:, i])       # sort all N by variable i
        sampled  = sort_idx[sub_idx]            # then take n_sub_out evenly spaced

        u_sorted = data[sampled, i]             # ch 0: sorted source
        for j in range(p):
            if i == j:
                continue
            v_sorted_by_u = data[sampled, j]    # ch 1: target sorted by source

            channels = [u_sorted, v_sorted_by_u]
            # kernel coeff channels (3)
            for cm in coeff_maps:
                channels.append(cm[(i, j)][sampled])
            # ANM residual channels (3): residuals of target j, sorted by source i
            for rm in resid_maps:
                channels.append(rm[j][sampled])

            out[i, j] = np.stack(channels, axis=0)  # [8, n_sub_out]

    return out


# ─── PC algorithm ─────────────────────────────────────────────────────────────

def compute_pc_features(data: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Run PC algorithm (Fisher-z CI test).
    Returns:
      pc_skel [n, n]: binary skeleton
      pc_dir  [n, n]: orientation; 1 if i→j, -1 if j→i, 0 otherwise
    """
    n = data.shape[1]
    zeros = np.zeros((n, n), dtype=np.float32)
    try:
        from causallearn.search.ConstraintBased.PC import pc as pc_search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cg = pc_search(data, alpha=alpha, indep_test="fisherz", verbose=False, show_progress=False)
        g = cg.G.graph  # [n, n]; entry [i,j]=-1,[j,i]=1 means i→j
        skel = (np.abs(g) > 0).astype(np.float32)
        directed = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if g[i, j] == 1 and g[j, i] == -1:   # i→j
                    directed[i, j] = 1.0
                elif g[i, j] == -1 and g[j, i] == 1:  # j→i
                    directed[i, j] = -1.0
        return skel, directed
    except Exception as e:
        logger.debug(f"PC failed (returning zeros): {e}")
        return zeros.copy(), zeros.copy()


# ─── DirectLiNGAM ─────────────────────────────────────────────────────────────

def compute_lingam_features(data: np.ndarray) -> np.ndarray:
    """
    Run DirectLiNGAM. Returns [n, n] causal adjacency matrix B where B[i,j]
    is the causal strength of the edge i→j.
    """
    n = data.shape[1]
    try:
        import lingam as lingam_lib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = lingam_lib.DirectLiNGAM()
            model.fit(data)
        B = np.abs(model.adjacency_matrix_).astype(np.float32)  # [n, n]
        return B
    except Exception:
        try:
            from causallearn.search.FCMBased.lingam import DirectLiNGAM as DLiNGAM
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = DLiNGAM()
                model.fit(data)
            return np.abs(model.adjacency_matrix_).astype(np.float32)
        except Exception as e2:
            logger.debug(f"LiNGAM failed (returning zeros): {e2}")
            return np.zeros((n, n), dtype=np.float32)


# ─── Node marginal statistics ──────────────────────────────────────────────────

def compute_node_stats(data: np.ndarray) -> np.ndarray:
    """Returns [n, 3]: (variance, skewness, kurtosis) per variable."""
    n = data.shape[1]
    stats = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        col = data[:, i]
        stats[i, 0] = float(np.var(col))
        stats[i, 1] = float(skew(col))
        stats[i, 2] = float(kurtosis(col))
    return stats


# ─── GraphData dataclass ──────────────────────────────────────────────────────

@dataclass
class GraphData:
    """All precomputed features for one observational graph."""
    cols: list[str]                    # variable names (includes X, Y)
    n: int

    # Edge sequences [n, n, 8, N_SUB]
    edge_seqs: np.ndarray

    # All-pairs MI [n, n] and conditional MI [n, n, n]
    pairwise_mi: np.ndarray
    cond_mi: np.ndarray

    # PC features
    pc_skel: np.ndarray               # [n, n]
    pc_dir: np.ndarray                # [n, n]

    # LiNGAM adjacency
    lingam_B: np.ndarray              # [n, n]

    # Node marginal stats [n, 3]
    node_stats: np.ndarray

    # Labels indexed by (x_name, y_name) → {v_name: class_idx}
    labels_by_xy: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)

    def get_node_features(self, x_name: str, y_name: str) -> np.ndarray:
        """
        Build [n, 7] node feature matrix for a given (X, Y) assignment.

        Features per node:
          0: I(v; X)            = 0 for X/Y nodes
          1: I(v; Y)
          2: I(v; X | Y)
          3: I(v; Y | X)
          4: variance
          5: skewness
          6: kurtosis
        """
        x_idx = self.cols.index(x_name)
        y_idx = self.cols.index(y_name)
        feats = np.zeros((self.n, 7), dtype=np.float32)
        feats[:, 4:7] = self.node_stats  # copy marginal stats for all nodes

        for v_idx, v_name in enumerate(self.cols):
            if v_name in (x_name, y_name):
                continue
            feats[v_idx, 0] = self.pairwise_mi[v_idx, x_idx]
            feats[v_idx, 1] = self.pairwise_mi[v_idx, y_idx]
            feats[v_idx, 2] = self.cond_mi[v_idx, x_idx, y_idx]
            feats[v_idx, 3] = self.cond_mi[v_idx, y_idx, x_idx]
        return feats

    def get_extra_edge_features(self) -> np.ndarray:
        """
        Scalar edge features (independent of XY assignment): [n, n, 3]
          partial correlation (pcorr), LiNGAM strength, PC skeleton, PC direction
        Actually returns [n, n, 3]: pcorr/LiNGAM are symmetric/asymmetric signals.
        We omit pcorr here (it's captured by CMI) and use lingam_B, pc_skel, pc_dir.
        """
        return np.stack([self.lingam_B, self.pc_skel, self.pc_dir], axis=-1)  # [n, n, 3]


# ─── Worker function ──────────────────────────────────────────────────────────

def _build_one_graph(args: tuple) -> Optional[GraphData]:
    """
    Worker: build GraphData from one (df, adj_df) pair.
    adj_df may be None for test samples.
    """
    df, adj_df = args
    try:
        cols = list(df.columns)
        n = len(cols)
        data = df.values.astype(np.float32)

        # 1. Kernel regression + ANM
        coeff_maps, resid_maps = compute_kernel_anm(data)
        edge_seqs = build_all_edge_sequences(data, coeff_maps, resid_maps)
        del coeff_maps, resid_maps

        # 2. MI / CMI
        pairwise_mi = compute_pairwise_mi(data)
        cond_mi = compute_cond_mi_tensor(data)

        # 3. PC
        pc_skel, pc_dir = compute_pc_features(data)

        # 4. LiNGAM
        lingam_B = compute_lingam_features(data)

        # 5. Node stats
        node_stats = compute_node_stats(data)

        gd = GraphData(
            cols=cols, n=n,
            edge_seqs=edge_seqs,
            pairwise_mi=pairwise_mi,
            cond_mi=cond_mi,
            pc_skel=pc_skel, pc_dir=pc_dir,
            lingam_B=lingam_B,
            node_stats=node_stats,
        )

        # 6. Labels (training only)
        if adj_df is not None:
            # Base (X, Y)
            gd.labels_by_xy[("X", "Y")] = get_all_node_labels(adj_df, cols)
            # XY remap augmentations
            for (x_name, y_name) in get_xy_remap_assignments(adj_df, cols):
                if (x_name, y_name) == ("X", "Y"):
                    continue
                gd.labels_by_xy[(x_name, y_name)] = relabel_for_xy(adj_df, cols, x_name, y_name)

        return gd
    except Exception as e:
        logger.warning(f"Failed to build graph ({list(df.columns)[:3]}...): {e}")
        return None


# ─── Main build function ──────────────────────────────────────────────────────

def build_graph_dataset(
    X_list: list[pd.DataFrame],
    y_list: Optional[list[pd.DataFrame]],
    output_dir: str,
    shard_size: int = 1000,
    n_workers: int = 48,
    tag: str = "v12",
) -> list[str]:
    """
    Process all graphs in parallel and write to shards.

    Args:
        X_list: list of observational DataFrames
        y_list: list of adjacency DataFrames (or None for test)
        output_dir: directory to write shard pickle files
        shard_size: number of GraphData objects per shard file
        n_workers: parallel CPU workers

    Returns: list of shard file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(X_list)
    y_list = y_list or [None] * n
    args = list(zip(X_list, y_list))

    print(f"Building {n} graph samples with {n_workers} workers...")
    results: list[Optional[GraphData]] = [None] * n
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=__import__("multiprocessing").get_context("fork")) as exe:
        futures = {exe.submit(_build_one_graph, a): i for i, a in enumerate(args)}
        for fut in tqdm(as_completed(futures), total=n, desc="Extracting features"):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                logger.warning(f"Graph {idx} failed: {e}")

    # Filter and write shards
    valid = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid)}/{n} graphs.")

    shard_paths: list[str] = []
    for shard_idx, start in enumerate(range(0, len(valid), shard_size)):
        chunk = valid[start:start + shard_size]
        path = os.path.join(output_dir, f"{tag}_shard_{shard_idx:04d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(chunk, f, protocol=4)
        shard_paths.append(path)
        print(f"  Wrote {path} ({len(chunk)} graphs)")
        del chunk
        gc.collect()

    return shard_paths


def build_from_pickles(
    x_train_path: str,
    y_train_path: str,
    output_dir: str,
    n_workers: int = 48,
    tag: str = "v12",
) -> list[str]:
    """Entry point: load CrunchDAO pickles, build and shard graph dataset."""
    print(f"Loading {x_train_path} ...")
    X_train: dict[str, pd.DataFrame] = pd.read_pickle(x_train_path)
    print(f"Loading {y_train_path} ...")
    y_train: dict[str, pd.DataFrame] = pd.read_pickle(y_train_path)

    names = list(X_train.keys())
    X_list = [X_train[k] for k in names]
    y_list = [y_train.get(k) for k in names]

    return build_graph_dataset(X_list, y_list, output_dir, n_workers=n_workers, tag=tag)


if __name__ == "__main__":
    paths = build_from_pickles(
        "data/X_train.pickle",
        "data/y_train.pickle",
        output_dir="dataset_cache/v12",
        n_workers=48,
    )
    print(f"Done. {len(paths)} shards written.")
