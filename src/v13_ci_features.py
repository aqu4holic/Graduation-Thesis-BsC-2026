"""
v13_ci_features.py — The "Everything" Model

Philosophy: The 8 causal role classes are DEFINED by conditional independence patterns.
The model should be given these patterns explicitly, not forced to learn them from raw curves.

Three information pathways:
  A. Edge-level curves (from v8b): 8ch conv1d → edge embedding
     - 2 sorted obs + 3 kernel coeff (bw 0.2/0.5/1.0) + 3 ANM residual (bw 0.2/0.5/1.0)
  B. Edge-level scalar stats (expanded from v5): per-edge statistical features → MLP
     - partial_corr, distance_corr, pearson, spearman, R²_fwd, R²_rev, hsic
  C. Node-level CI features (NEW — the key innovation): per-node features → MLP
     - CMI(v,X|Y), CMI(v,Y|X), CMI(X,Y|v) via k-NN
     - HSIC(v,X), HSIC(v,Y), HSIC(X,Y)
     - R²(v→X), R²(X→v), R²(v→Y), R²(Y→v) via kernel regression
     - corr(v,X), corr(v,Y), corr(X,Y)
     - partial_corr(v,X|rest), partial_corr(v,Y|rest), partial_corr(X,Y|rest)
     - dimension (number of variables)

Architecture:
  1. Pathway A → (B,E,d) edge embedding via conv1d
  2. Pathway B → (B,E,d) stat embedding via MLP
  3. Merge [A, B, type_emb] → (B,E,d) with structural attention bias (v11)
  4. 2-3× SelfAttention with structural bias
  5. Node head: gather 4 edges + inject Pathway C node features → 8-class

Loss: Focal loss (γ=2) with inverse-frequency weighting + label smoothing.

Usage:
    python v13_ci_features.py
"""

import typing
import os
import sys
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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

# === Channel config (same as v8b) ===
BANDWIDTHS = [0.2, 0.5, 1.0]
N_CHANNELS = 2 + len(BANDWIDTHS) + len(BANDWIDTHS)  # 2 sorted + 3 kernel + 3 ANM = 8

# === Edge-level scalar stats ===
N_EDGE_STATS = 7  # partial_corr, dist_corr, pearson, spearman, R2_fwd, R2_rev, hsic

# === Node-level CI features ===
N_NODE_FEATURES = 17  # see compute_node_ci_features

# === Structural attention bias (from v11) ===
N_STRUCT_BIAS_TYPES = 6

# === Training ===
MAX_EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
AUG_NOISE_STD = 0.01

LOCAL_CACHE_DIR = "dataset_cache/"
CACHE_TAG = "v13_ci"


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
        nx.DiGraph([("X", "Y"), ("v", "Y"), ("v", "X")]): "Cause of Y",  # dup handled by label
        nx.DiGraph([("X", "Y")]): "Independent",
    }
    adjacency_label = {}
    nodelist = ["v", "X", "Y"]
    for G, label in graph_label.items():
        # key = graph_nodes_representation(G, ["v", "X", "Y"])
        g = G.copy()
        for node in nodelist:
            if node not in g:
                g.add_node(node)
        mat = nx.to_numpy_array(g, nodelist=nodelist)
        key = tuple(mat.flatten())
        adjacency_label[key] = CLASS_NAMES.index(label)
    return adjacency_label

_ADJACENCY_LABEL = create_graph_label()

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
# Edge type encoding
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
# Feature Computation: Kernel Regression (from v8b)
# ============================================================
def compute_multivariate_kernel_coefficients(
    data: np.ndarray, n_sub: int = None, bandwidth: float = 0.5,
) -> tuple:
    """
    Returns (coeff_map, resid_map) where:
      coeff_map[(k,j)][i] = kernel regression coefficient c_{i, k→j}
      resid_map[j][i] = residual of predicting j from all others at obs i
    """
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
    all_predictions = {}
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
        # Predicted values at subsample points
        y_hat_sub = np.einsum('ia,la,il->i', c_all, X_design, W) / (W.sum(axis=1) + 1e-10)
        all_predictions[j] = y_hat_sub

    # NN interpolation to all N points
    dist_to_sub = np.sum((data[:, None, :] - data_sub[None, :, :]) ** 2, axis=-1)
    nearest = np.argmin(dist_to_sub, axis=1)

    coeff_map = {}
    for j in range(p):
        c_all, other_cols = all_coeffs[j]
        for idx_in_other, k in enumerate(other_cols):
            coeff_sub = c_all[:, idx_in_other + 1]
            coeff_map[(k, j)] = coeff_sub[nearest].astype(np.float32)

    # Residual map: resid[j] = data[:,j] - y_hat_j (interpolated via NN)
    resid_map = {}
    for j in range(p):
        c_all, other_cols = all_coeffs[j]
        c_nn = c_all[nearest]  # (N, p)
        X_full = np.concatenate([np.ones((N, 1)), data[:, [k for k in range(p) if k != j]]], axis=1)
        y_hat = (c_nn * X_full).sum(axis=1)
        resid_map[j] = (data[:, j] - y_hat).astype(np.float32)

    return coeff_map, resid_map


# ============================================================
# Feature Computation: Edge-level scalar statistics
# ============================================================
def _hsic_statistic(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Compute HSIC with Gaussian kernel between 1D arrays."""
    n = len(x)
    if n < 5:
        return 0.0
    # Subsample for speed
    if n > 500:
        idx = np.random.choice(n, 500, replace=False)
        x, y = x[idx], y[idx]
        n = 500

    Kx = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / (sigma ** 2 + 1e-10))
    Ky = np.exp(-0.5 * (y[:, None] - y[None, :]) ** 2 / (sigma ** 2 + 1e-10))

    # Center the kernel matrices
    H = np.eye(n) - 1.0 / n
    Kxc = H @ Kx @ H
    Kyc = H @ Ky @ H

    hsic = np.trace(Kxc @ Kyc) / (n ** 2)
    return float(hsic)


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute distance correlation between 1D arrays."""
    n = len(x)
    if n < 5:
        return 0.0
    # Subsample for speed
    if n > 500:
        idx = np.random.choice(n, 500, replace=False)
        x, y = x[idx], y[idx]
        n = 500

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()

    denom = np.sqrt(dvar_x * dvar_y)
    if denom < 1e-10:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom))


def _r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """R² of predicting y from x via linear regression."""
    if np.var(x) < 1e-10 or np.var(y) < 1e-10:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr ** 2)


def compute_edge_statistics(data: np.ndarray) -> np.ndarray:
    """
    Compute per-edge scalar statistics.

    Returns:
        stat_matrix: (p, p, N_EDGE_STATS) where stat_matrix[i,j] contains stats for edge i→j
        Features: [partial_corr, dist_corr, pearson, spearman, R2_fwd, R2_rev, hsic]
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

    # Median absolute deviation for HSIC bandwidth
    sigma = np.median(np.abs(data - np.median(data, axis=0, keepdims=True)), axis=0)
    sigma = np.maximum(sigma, 0.1)

    stat_matrix = np.zeros((p, p, N_EDGE_STATS), dtype=np.float32)

    for i in range(p):
        xi = data[:, i]
        for j in range(p):
            if i == j:
                continue
            xj = data[:, j]

            # 0: Partial correlation
            stat_matrix[i, j, 0] = pcorr[i, j]

            # 1: Distance correlation
            stat_matrix[i, j, 1] = _distance_correlation(xi, xj)

            # 2: Pearson correlation
            pc = np.corrcoef(xi, xj)[0, 1]
            stat_matrix[i, j, 2] = pc if not np.isnan(pc) else 0.0

            # 3: Spearman correlation
            sc, _ = scipy_stats.spearmanr(xi, xj)
            stat_matrix[i, j, 3] = sc if not np.isnan(sc) else 0.0

            # 4: R² forward (i predicts j)
            stat_matrix[i, j, 4] = _r_squared(xi, xj)

            # 5: R² reverse (j predicts i)
            stat_matrix[i, j, 5] = _r_squared(xj, xi)

            # 6: HSIC
            stat_matrix[i, j, 6] = _hsic_statistic(xi, xj, sigma=np.mean(sigma[[i, j]]))

    return stat_matrix


# ============================================================
# Feature Computation: Node-level CI features (THE KEY INNOVATION)
# ============================================================
def _knn_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = 7) -> float:
    """
    Estimate conditional mutual information I(X;Y|Z) using k-NN (KSG-style).

    Uses the Frenzel-Pompe estimator:
        I(X;Y|Z) = ψ(k) - <ψ(n_xz + 1) + ψ(n_yz + 1) - ψ(n_z + 1)>

    where n_xz, n_yz, n_z are neighbor counts in the projected subspaces
    within the distance to the k-th neighbor in the full (X,Y,Z) space.

    x, y: (N,) arrays
    z: (N,) or (N, d) array for conditioning
    """
    from scipy.special import digamma
    from scipy.spatial import cKDTree

    n = len(x)
    if n < k + 5:
        return 0.0

    # Subsample for speed
    max_n = 500
    if n > max_n:
        idx = np.random.choice(n, max_n, replace=False)
        x, y, z = x[idx], y[idx], z[idx] if z.ndim == 1 else z[idx]
        n = max_n

    # Standardize
    x = (x - x.mean()) / (x.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)
    if z.ndim == 1:
        z = (z - z.mean()) / (z.std() + 1e-10)
        z = z[:, None]
    else:
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-10)

    x = x[:, None]  # (n,1)
    y = y[:, None]  # (n,1)

    # Joint space: (X, Y, Z)
    xyz = np.hstack([x, y, z])
    xz = np.hstack([x, z])
    yz = np.hstack([y, z])

    # Build KD-trees with Chebyshev (max) metric
    tree_xyz = cKDTree(xyz)
    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z)

    # Find k-th neighbor distance in joint space
    dd, _ = tree_xyz.query(xyz, k=k + 1, p=np.inf)
    eps = dd[:, -1]  # distance to k-th neighbor (excluding self)

    # Count neighbors within eps in each subspace
    n_xz = np.array([tree_xz.query_ball_point(xz[i], eps[i], p=np.inf, return_length=True) - 1
                      for i in range(n)])
    n_yz = np.array([tree_yz.query_ball_point(yz[i], eps[i], p=np.inf, return_length=True) - 1
                      for i in range(n)])
    n_z = np.array([tree_z.query_ball_point(z[i], eps[i], p=np.inf, return_length=True) - 1
                     for i in range(n)])

    # Clamp to at least 1
    n_xz = np.maximum(n_xz, 1)
    n_yz = np.maximum(n_yz, 1)
    n_z = np.maximum(n_z, 1)

    cmi = digamma(k) - np.mean(digamma(n_xz) + digamma(n_yz) - digamma(n_z))
    return float(max(cmi, 0.0))  # CMI >= 0 in theory


def _knn_mi(x: np.ndarray, y: np.ndarray, k: int = 7) -> float:
    """Estimate mutual information I(X;Y) using KSG estimator."""
    from scipy.special import digamma
    from scipy.spatial import cKDTree

    n = len(x)
    if n < k + 5:
        return 0.0

    max_n = 500
    if n > max_n:
        idx = np.random.choice(n, max_n, replace=False)
        x, y = x[idx], y[idx]
        n = max_n

    x = (x - x.mean()) / (x.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)
    x = x[:, None]
    y = y[:, None]

    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    dd, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dd[:, -1]

    n_x = np.array([tree_x.query_ball_point(x[i], eps[i], p=np.inf, return_length=True) - 1
                     for i in range(n)])
    n_y = np.array([tree_y.query_ball_point(y[i], eps[i], p=np.inf, return_length=True) - 1
                     for i in range(n)])

    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)

    mi = digamma(k) + digamma(n) - np.mean(digamma(n_x) + digamma(n_y))
    return float(max(mi, 0.0))


def compute_node_ci_features(data: np.ndarray, cols: list) -> np.ndarray:
    """
    Compute per-node CI features for all non-X/Y variables.

    For each variable v (not X or Y), compute:
      0: CMI(v, X | Y)        — does v carry info about X beyond Y?
      1: CMI(v, Y | X)        — does v carry info about Y beyond X?
      2: CMI(X, Y | v)        — does X-Y dependence change when conditioning on v?
      3: MI(v, X)              — marginal dependence v-X
      4: MI(v, Y)              — marginal dependence v-Y
      5: MI(X, Y)              — marginal dependence X-Y (same for all v, context feature)
      6: HSIC(v, X)
      7: HSIC(v, Y)
      8: corr(v, X)
      9: corr(v, Y)
     10: corr(X, Y)           — context
     11: partial_corr(v, X | rest)
     12: partial_corr(v, Y | rest)
     13: partial_corr(X, Y | rest)   — context
     14: R²(v→X) - R²(X→v)   — direction asymmetry v-X
     15: R²(v→Y) - R²(Y→v)   — direction asymmetry v-Y
     16: log(p)                — graph dimension (log for scale invariance)

    Returns:
        node_features: (K, N_NODE_FEATURES) where K = number of non-X/Y variables
        node_names: list of variable names in order
    """
    N, p = data.shape
    col_idx = {name: i for i, name in enumerate(cols)}
    x_idx = col_idx["X"]
    y_idx = col_idx["Y"]

    x_data = data[:, x_idx]
    y_data = data[:, y_idx]

    # Precompute partial correlations
    cov = np.cov(data.T)
    cov += 1e-6 * np.eye(p)
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        precision = np.eye(p)
    diag = np.sqrt(np.maximum(np.diag(precision), 1e-10))
    pcorr_matrix = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcorr_matrix, 0.0)

    # Precompute MI(X, Y), corr(X, Y), etc. (shared across all nodes)
    mi_xy = _knn_mi(x_data, y_data)
    corr_xy = np.corrcoef(x_data, y_data)[0, 1]
    if np.isnan(corr_xy):
        corr_xy = 0.0
    pcorr_xy = pcorr_matrix[x_idx, y_idx]

    # HSIC bandwidth
    sigma = np.median(np.abs(data - np.median(data, axis=0, keepdims=True)), axis=0)
    sigma = np.maximum(sigma, 0.1)

    non_xy_vars = [c for c in cols if c not in ("X", "Y")]
    K = len(non_xy_vars)
    node_features = np.zeros((K, N_NODE_FEATURES), dtype=np.float32)
    node_names = non_xy_vars

    for vi, v_name in enumerate(non_xy_vars):
        v_idx = col_idx[v_name]
        v_data = data[:, v_idx]

        # CMI features (the most important ones)
        node_features[vi, 0] = _knn_cmi(v_data, x_data, y_data, k=7)     # CMI(v,X|Y)
        node_features[vi, 1] = _knn_cmi(v_data, y_data, x_data, k=7)     # CMI(v,Y|X)
        node_features[vi, 2] = _knn_cmi(x_data, y_data, v_data, k=7)     # CMI(X,Y|v)

        # MI features
        node_features[vi, 3] = _knn_mi(v_data, x_data)                    # MI(v,X)
        node_features[vi, 4] = _knn_mi(v_data, y_data)                    # MI(v,Y)
        node_features[vi, 5] = mi_xy                                      # MI(X,Y)

        # HSIC features
        sig_v = sigma[v_idx]
        node_features[vi, 6] = _hsic_statistic(v_data, x_data, sigma=0.5 * (sig_v + sigma[x_idx]))
        node_features[vi, 7] = _hsic_statistic(v_data, y_data, sigma=0.5 * (sig_v + sigma[y_idx]))

        # Correlation features
        cx = np.corrcoef(v_data, x_data)[0, 1]
        cy = np.corrcoef(v_data, y_data)[0, 1]
        node_features[vi, 8] = cx if not np.isnan(cx) else 0.0
        node_features[vi, 9] = cy if not np.isnan(cy) else 0.0
        node_features[vi, 10] = corr_xy

        # Partial correlation features
        node_features[vi, 11] = pcorr_matrix[v_idx, x_idx]
        node_features[vi, 12] = pcorr_matrix[v_idx, y_idx]
        node_features[vi, 13] = pcorr_xy

        # Direction asymmetry
        r2_vx = _r_squared(v_data, x_data)
        r2_xv = _r_squared(x_data, v_data)
        r2_vy = _r_squared(v_data, y_data)
        r2_yv = _r_squared(y_data, v_data)
        node_features[vi, 14] = r2_vx - r2_xv  # >0 suggests v→X
        node_features[vi, 15] = r2_vy - r2_yv  # >0 suggests v→Y

        # Graph dimension
        node_features[vi, 16] = np.log(p)

    return node_features, node_names


# ============================================================
# Feature Computation: Structural Bias (from v11)
# ============================================================
def compute_structural_bias(cols: list) -> np.ndarray:
    """
    Compute structural attention bias matrix based on edge topology.

    Bias types encode which pairs of edges should attend more to each other:
      0: same source (u→a, u→b)
      1: same target (a→v, b→v)
      2: chain (a→b exists as another edge, and current pair forms chain)
      3: involves same X/Y node
      4: reverse edges (u→v, v→u)
      5: default (no special relationship)

    Returns: (E, E) int array of bias type indices
    """
    p = len(cols)
    E = p * (p - 1)
    edge_list = []
    for i, u in enumerate(cols):
        for j, v in enumerate(cols):
            if i != j:
                edge_list.append((i, j, u, v))

    bias_types = np.full((E, E), 5, dtype=np.int64)  # default type

    for ei, (si, ti, su, tu) in enumerate(edge_list):
        for ej, (sj, tj, suj, tuj) in enumerate(edge_list):
            if ei == ej:
                bias_types[ei, ej] = 0
                continue

            # Same source
            if si == sj:
                bias_types[ei, ej] = 0
            # Same target
            elif ti == tj:
                bias_types[ei, ej] = 1
            # Reverse edge
            elif si == tj and ti == sj:
                bias_types[ei, ej] = 4
            # Chain: target of one is source of other
            elif ti == sj or tj == si:
                bias_types[ei, ej] = 2
            # Involves same X or Y
            elif (su in ("X", "Y") and suj in ("X", "Y")) or \
                 (tu in ("X", "Y") and tuj in ("X", "Y")) or \
                 (su in ("X", "Y") and tuj in ("X", "Y")) or \
                 (tu in ("X", "Y") and suj in ("X", "Y")):
                bias_types[ei, ej] = 3

    return bias_types


# ============================================================
# Data Preprocessing: Build all features for one sample
# ============================================================
def build_edge_tensor(df: pd.DataFrame) -> tuple:
    """
    Build 8-channel edge tensor + edge stats + node CI features + structural bias.

    Edge channels (same as v8b):
      Ch 0: sorted u observations
      Ch 1: v observations sorted by u
      Ch 2-4: kernel regression coeff at bw 0.2, 0.5, 1.0
      Ch 5-7: ANM residual at bw 0.2, 0.5, 1.0

    Returns:
      edge_data:       (E, N_CHANNELS, N) float32
      edge_types:      (E,) int64
      edge_stats:      (E, N_EDGE_STATS) float32
      node_features:   (K, N_NODE_FEATURES) float32
      node_names:      list of K variable names
      struct_bias:     (E, E) int64
    """
    cols = list(df.columns)
    p = len(cols)
    data = df.values.astype(np.float32)
    N = data.shape[0]

    # Share the same subsample across bandwidths
    n_sub = min(N_KERNEL, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)

    # Compute kernel regression + ANM residuals at each bandwidth
    coeff_maps = []
    resid_maps = []
    for bw in BANDWIDTHS:
        cm, rm = compute_multivariate_kernel_coefficients(data, n_sub=n_sub, bandwidth=bw)
        coeff_maps.append(cm)
        resid_maps.append(rm)

    # Compute edge-level statistics
    stat_matrix = compute_edge_statistics(data)

    # Compute node-level CI features
    node_features, node_names = compute_node_ci_features(data, cols)

    # Compute structural bias
    struct_bias = compute_structural_bias(cols)

    # Build edge tensors
    edges = []
    edge_types_list = []
    edge_stats_list = []

    for i, u_name in enumerate(cols):
        sort_idx = np.argsort(data[:, i])
        u_sorted = data[sort_idx, i]

        for j, v_name in enumerate(cols):
            if i == j:
                continue

            v_sorted_by_u = data[sort_idx, j]

            # 8 channels: sorted_u, sorted_v, 3×kernel_coeff, 3×ANM_resid
            channels = [u_sorted, v_sorted_by_u]

            # Kernel regression coefficients at 3 bandwidths
            for cm in coeff_maps:
                channels.append(cm[(i, j)][sort_idx])

            # ANM residuals at 3 bandwidths (residual of TARGET j, sorted by SOURCE i)
            for rm in resid_maps:
                channels.append(rm[j][sort_idx])

            edge_tensor = np.stack(channels, axis=0)  # (8, N)
            edges.append(edge_tensor)
            edge_types_list.append(_edge_type(u_name, v_name))

            # Edge-level scalar stats
            edge_stats_list.append(stat_matrix[i, j])

    edge_data = np.stack(edges, axis=0).astype(np.float32)         # (E, 8, N)
    edge_types = np.array(edge_types_list, dtype=np.int64)          # (E,)
    edge_stats = np.stack(edge_stats_list, axis=0).astype(np.float32)  # (E, 7)

    return edge_data, edge_types, edge_stats, node_features, node_names, struct_bias


def _build_single(args):
    """Worker function for parallel preprocessing."""
    df, y_df = args
    result = {}

    edge_data, edge_types, edge_stats, node_features, node_names, struct_bias = build_edge_tensor(df)
    cols = list(df.columns)
    p = len(cols)

    result["edge_data"] = edge_data
    result["edge_types"] = edge_types
    result["edge_stats"] = edge_stats
    result["node_features"] = node_features
    result["node_names"] = node_names
    result["struct_bias"] = struct_bias
    result["cols"] = cols
    result["p"] = p

    if y_df is not None:
        adj_np = y_df.values.astype(np.float32)
        adj_cols = list(y_df.columns)
        result["adj"] = adj_np
        result["adj_cols"] = adj_cols

        # Precompute edge labels
        edge_labels = []
        for i, u_name in enumerate(cols):
            for j, v_name in enumerate(cols):
                if i == j:
                    continue
                u_idx = adj_cols.index(u_name)
                v_idx = adj_cols.index(v_name)
                edge_labels.append(int(adj_np[u_idx, v_idx]))
        result["edge_labels"] = np.array(edge_labels, dtype=np.int64)

        # Precompute node labels
        adjacency_label = _ADJACENCY_LABEL
        adj_df = pd.DataFrame(adj_np, index=adj_cols, columns=adj_cols)
        labels = get_labels(adj_df, adjacency_label)
        node_labels = np.array([labels[v] for v in node_names], dtype=np.int64)
        result["node_labels"] = node_labels

    return result


# ============================================================
# Dataset
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
    max_K = max(item["node_features"].shape[0] for item in batch)
    N = batch[0]["edge_data"].shape[2]
    C = batch[0]["edge_data"].shape[1]

    edge_data = torch.zeros(B, max_E, C, N)
    edge_types = torch.zeros(B, max_E, dtype=torch.long)
    edge_mask = torch.zeros(B, max_E, dtype=torch.bool)
    edge_stats = torch.zeros(B, max_E, N_EDGE_STATS)
    node_features = torch.zeros(B, max_K, N_NODE_FEATURES)
    node_mask = torch.zeros(B, max_K, dtype=torch.bool)
    struct_bias = torch.full((B, max_E, max_E), N_STRUCT_BIAS_TYPES - 1, dtype=torch.long)  # default type

    has_labels = False
    edge_labels = torch.zeros(B, max_E, dtype=torch.long)
    node_labels = torch.zeros(B, max_K, dtype=torch.long)

    cols_list = []

    for b, item in enumerate(batch):
        E = item["edge_data"].shape[0]
        K = item["node_features"].shape[0]

        edge_data[b, :E] = torch.from_numpy(item["edge_data"])
        edge_types[b, :E] = torch.from_numpy(item["edge_types"])
        edge_mask[b, :E] = True
        edge_stats[b, :E] = torch.from_numpy(item["edge_stats"])
        node_features[b, :K] = torch.from_numpy(item["node_features"])
        node_mask[b, :K] = True
        struct_bias[b, :E, :E] = torch.from_numpy(item["struct_bias"])
        cols_list.append(item["cols"])

        if "edge_labels" in item:
            has_labels = True
            edge_labels[b, :E] = torch.from_numpy(item["edge_labels"])
            node_labels[b, :K] = torch.from_numpy(item["node_labels"])

    out = {
        "edge_data": edge_data,
        "edge_types": edge_types,
        "edge_mask": edge_mask,
        "edge_stats": edge_stats,
        "node_features": node_features,
        "node_mask": node_mask,
        "struct_bias": struct_bias,
        "cols": cols_list,
    }
    if has_labels:
        out["edge_labels"] = edge_labels
        out["node_labels"] = node_labels

    return out


# ============================================================
# Model Components
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
        n_channels = n_channels or N_CHANNELS
        self.linear = nn.Linear(n_channels, d)

    def forward(self, x):
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class EdgeFeatureExtractor(nn.Module):
    def __init__(self, d=64, n_blocks=5):
        super().__init__()
        self.stem = StemLayer(d)
        self.blocks = nn.Sequential(*[ConvBlock(d) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.pool(x).squeeze(-1)


class StatProjector(nn.Module):
    """Project per-edge scalar statistics into d-dim."""
    def __init__(self, n_stats, d):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_stats, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, stats):
        return self.proj(stats)


class NodeFeatureProjector(nn.Module):
    """Project per-node CI features into d-dim."""
    def __init__(self, n_features, d):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, 2 * d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
        )

    def forward(self, features):
        return self.proj(features)


class SelfAttentionWithBias(nn.Module):
    """Self-attention with learned structural bias (from v11)."""
    def __init__(self, d=64, n_heads=4, n_bias_types=N_STRUCT_BIAS_TYPES):
        super().__init__()
        self.n_heads = n_heads
        self.d = d
        self.head_dim = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

        # Learned bias per head per structural type
        self.bias_emb = nn.Embedding(n_bias_types, n_heads)

        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, struct_bias=None, key_padding_mask=None):
        B, E, d = x.shape
        h = self.n_heads
        hd = self.head_dim

        q = self.q_proj(x).view(B, E, h, hd).transpose(1, 2)  # (B, h, E, hd)
        k = self.k_proj(x).view(B, E, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, E, h, hd).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)  # (B, h, E, E)

        # Add structural bias
        if struct_bias is not None:
            bias = self.bias_emb(struct_bias)  # (B, E, E, h)
            bias = bias.permute(0, 3, 1, 2)   # (B, h, E, E)
            attn = attn + bias

        # Apply padding mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, E)
                float('-inf')
            )

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, h, E, hd)
        out = out.transpose(1, 2).contiguous().view(B, E, d)
        out = self.o_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance on hard examples."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing via standard CE
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================
# Full Model
# ============================================================
class ADIAModel(nn.Module):
    """
    v13: Three-pathway model with CI features and structural bias.

    Pathway A: 8-channel conv1d → edge embedding
    Pathway B: Edge scalar stats → MLP embedding
    Pathway C: Node CI features → MLP embedding (injected at node head)

    Pipeline:
      1. Conv: (B,E,8,N) → (B,E,d)
      2. StatProj: (B,E,7) → (B,E,d)
      3. Merge [conv, stat, type_emb] → (B,E,d)
      4. 3× SelfAttention with structural bias
      5. Edge head: (B,E,d) → (B,E,2)
      6. Node head: gather 4 edges + NodeFeatureProj → 8-class
    """
    def __init__(self, d=None, n_edge_types=None, n_attn_layers=3,
                 aug_noise_std=AUG_NOISE_STD):
        super().__init__()
        d = d or D_MODEL
        n_edge_types = n_edge_types or N_EDGE_TYPES
        self.d = d
        self.aug_noise_std = aug_noise_std

        # Pathway A: Conv on 8-channel edge tensors
        self.extractor = EdgeFeatureExtractor(d, n_blocks=5)

        # Pathway B: Edge scalar stats
        self.stat_proj = StatProjector(N_EDGE_STATS, d)

        # Edge type embedding
        self.edge_type_emb = nn.Embedding(n_edge_types, d)

        # 3-input merge: conv + stat + type
        self.edge_merge = MergeOperator(n_inputs=3, d=d)

        # Self-attention with structural bias
        self.attn_layers = nn.ModuleList([
            SelfAttentionWithBias(d, n_heads=4) for _ in range(n_attn_layers)
        ])

        # Edge head (binary adjacency, training auxiliary)
        self.edge_head = nn.Linear(d, 2)

        # Pathway C: Node CI features
        self.node_feat_proj = NodeFeatureProjector(N_NODE_FEATURES, d)

        # Node head: 4 edge embeddings + 1 node feature embedding = 5 inputs
        self.node_merge = MergeOperator(n_inputs=5, d=d)
        self.node_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, N_CLASSES),
        )

    def forward(self, edge_data, edge_types, edge_mask, cols_list,
                edge_stats=None, node_features=None, node_mask=None,
                struct_bias=None):
        B, E, C, N = edge_data.shape

        # --- Training augmentation ---
        if self.training and self.aug_noise_std > 0:
            edge_data = edge_data + torch.randn_like(edge_data) * self.aug_noise_std

        # --- Pathway A: Conv edge features ---
        edge_emb = self.extractor(edge_data.view(B * E, C, N)).view(B, E, self.d)

        # --- Pathway B: Edge statistics ---
        if edge_stats is not None:
            stat_emb = self.stat_proj(edge_stats)
        else:
            stat_emb = torch.zeros_like(edge_emb)

        # --- Merge with edge type ---
        type_emb = self.edge_type_emb(edge_types)
        edge_emb = self.edge_merge([edge_emb, stat_emb, type_emb])

        # --- Self-attention with structural bias ---
        pad_mask = ~edge_mask
        for attn in self.attn_layers:
            edge_emb = attn(edge_emb, struct_bias=struct_bias, key_padding_mask=pad_mask)

        # --- Edge head ---
        edge_logits = self.edge_head(edge_emb)

        # --- Node head ---
        # Pathway C: project node CI features
        if node_features is not None:
            node_feat_emb = self.node_feat_proj(node_features)  # (B, K, d)
        else:
            node_feat_emb = None

        node_logits_list = []
        for b in range(B):
            cols = cols_list[b]
            p = len(cols)
            col_idx = {name: i for i, name in enumerate(cols)}

            # Build edge index map
            edge_order = {}
            idx = 0
            for i in range(p):
                for j in range(p):
                    if i != j:
                        edge_order[(cols[i], cols[j])] = idx
                        idx += 1

            non_xy = [c for c in cols if c not in ("X", "Y")]
            sample_node_logits = []

            for vi, v_name in enumerate(non_xy):
                # Gather 4 edges
                e_vx = edge_order.get((v_name, "X"), None)
                e_vy = edge_order.get((v_name, "Y"), None)
                e_xv = edge_order.get(("X", v_name), None)
                e_yv = edge_order.get(("Y", v_name), None)

                edge_indices = [e_vx, e_vy, e_xv, e_yv]
                gathered = []
                for ei in edge_indices:
                    if ei is not None and ei < E:
                        gathered.append(edge_emb[b, ei])
                    else:
                        gathered.append(torch.zeros(self.d, device=edge_emb.device))

                # Add node CI feature embedding
                if node_feat_emb is not None and vi < node_feat_emb.shape[1]:
                    nf = node_feat_emb[b, vi]
                else:
                    nf = torch.zeros(self.d, device=edge_emb.device)

                merged = self.node_merge(gathered + [nf])
                logits = self.node_head(merged)
                sample_node_logits.append(logits)

            if sample_node_logits:
                node_logits_list.append(torch.stack(sample_node_logits))
            else:
                node_logits_list.append(torch.zeros(0, N_CLASSES, device=edge_emb.device))

        return edge_logits, node_logits_list


# ============================================================
# Lightning Module
# ============================================================
class ADIALightningModule(pl.LightningModule):
    def __init__(self, class_weights=None, lr=LR):
        super().__init__()
        self.model = ADIAModel()
        self.lr = lr

        # Edge loss: standard CE
        self.edge_loss_fn = nn.CrossEntropyLoss()

        # Node loss: Focal loss with class weights
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
        else:
            w = None
        self.node_loss_fn = FocalLoss(
            gamma=FOCAL_GAMMA,
            weight=w,
            label_smoothing=LABEL_SMOOTHING,
        )

    def forward(self, batch):
        return self.model(
            batch["edge_data"], batch["edge_types"], batch["edge_mask"],
            batch["cols"],
            edge_stats=batch.get("edge_stats"),
            node_features=batch.get("node_features"),
            node_mask=batch.get("node_mask"),
            struct_bias=batch.get("struct_bias"),
        )

    def _compute_loss(self, batch, prefix="train"):
        edge_logits, node_logits_list = self(batch)

        # Edge loss
        edge_mask = batch["edge_mask"]
        edge_labels = batch["edge_labels"]
        valid_edges = edge_mask.view(-1)
        edge_loss = self.edge_loss_fn(
            edge_logits.view(-1, 2)[valid_edges],
            edge_labels.view(-1)[valid_edges],
        )

        # Node loss
        all_node_logits = []
        all_node_labels = []
        B = len(node_logits_list)
        for b in range(B):
            nl = node_logits_list[b]
            K = nl.shape[0]
            if K > 0:
                all_node_logits.append(nl)
                all_node_labels.append(batch["node_labels"][b, :K])

        if all_node_logits:
            all_node_logits = torch.cat(all_node_logits, dim=0)
            all_node_labels = torch.cat(all_node_labels, dim=0)
            node_loss = self.node_loss_fn(all_node_logits, all_node_labels)

            # Per-class accuracy logging
            if prefix == "val":
                preds = all_node_logits.argmax(dim=-1)
                for c in range(N_CLASSES):
                    mask = all_node_labels == c
                    if mask.sum() > 0:
                        acc = (preds[mask] == c).float().mean()
                        self.log(f"{prefix}_acc_{CLASS_NAMES[c]}", acc, prog_bar=False, batch_size=B)

                overall_acc = (preds == all_node_labels).float().mean()
                self.log(f"{prefix}_acc", overall_acc, prog_bar=True, batch_size=B)
        else:
            node_loss = torch.tensor(0.0, device=self.device)

        total_loss = edge_loss + node_loss
        self.log(f"{prefix}_loss", total_loss, prog_bar=True, batch_size=B)
        self.log(f"{prefix}_edge_loss", edge_loss, prog_bar=False, batch_size=B)
        self.log(f"{prefix}_node_loss", node_loss, prog_bar=False, batch_size=B)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
        )
        return [optimizer], [scheduler]


# ============================================================
# Data Loading & Caching
# ============================================================
def build_dataset(X_train, y_train, cache_path=None, n_workers=16):
    """Build preprocessed dataset with caching."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            items = pickle.load(f)
        print(f"  Loaded {len(items)} items")
        return items

    print(f"Building dataset ({len(X_train)} samples)...")
    args_list = []
    for key in tqdm(X_train.keys(), desc="Preparing"):
        df = X_train[key]
        y_df = y_train[key] if y_train is not None else None
        args_list.append((df, y_df))

    items = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=__import__('multiprocessing').get_context('fork')) as pool:
        futures = {pool.submit(_build_single, args): i for i, args in enumerate(args_list)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                items.append(result)
            except Exception as e:
                print(f"Error: {e}")

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
    """Run inference on a dataset and return predictions."""
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    all_predictions = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring"):
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            edge_logits, node_logits_list = model(
                batch["edge_data"], batch["edge_types"], batch["edge_mask"],
                batch["cols"],
                edge_stats=batch.get("edge_stats"),
                node_features=batch.get("node_features"),
                node_mask=batch.get("node_mask"),
                struct_bias=batch.get("struct_bias"),
            )

            B = len(node_logits_list)
            for b in range(B):
                cols = batch["cols"][b]
                non_xy = [c for c in cols if c not in ("X", "Y")]
                nl = node_logits_list[b]
                preds = nl.argmax(dim=-1).cpu().numpy()
                for vi, v_name in enumerate(non_xy):
                    all_predictions[(b, v_name)] = int(preds[vi])

    return all_predictions


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", help="Directory with pickle files")
    parser.add_argument("--cache_dir", default=LOCAL_CACHE_DIR)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n_workers", type=int, default=os.cpu_count() - 1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    X_train = pd.read_pickle(os.path.join(args.data_dir, "X_train.pickle"))
    y_train = pd.read_pickle(os.path.join(args.data_dir, "y_train.pickle"))
    X_test = pd.read_pickle(os.path.join(args.data_dir, "X_test_reduced.pickle"))

    # Build datasets
    cache_path = os.path.join(args.cache_dir, f"train_dataset_{CACHE_TAG}_nk{N_KERNEL}.pkl")
    all_items = build_dataset(X_train, y_train, cache_path=cache_path, n_workers=args.n_workers)

    # Train/val split
    train_items, val_items = train_test_split(all_items, test_size=0.1, random_state=42)
    print(f"Train: {len(train_items)}, Val: {len(val_items)}")

    train_dataset = CausalEdgeDataset(train_items)
    val_dataset = CausalEdgeDataset(val_items)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Compute class weights
    all_labels = []
    for item in all_items:
        if "node_labels" in item:
            all_labels.extend(item["node_labels"].tolist())
    label_counts = np.bincount(all_labels, minlength=N_CLASSES).astype(np.float32)
    class_weights = 1.0 / (label_counts + 1.0)
    class_weights = class_weights / class_weights.sum() * N_CLASSES
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.round(3)))}")

    # Train
    model = ADIALightningModule(class_weights=class_weights, lr=args.lr)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision="32-true",
        gradient_clip_val=1.0,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc", mode="max", save_top_k=1,
                filename="v13-{epoch}-{val_acc:.4f}",
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_acc", mode="max", patience=10,
            ),
        ],
        log_every_n_steps=10,
    )

    if not args.eval_only:
        trainer.fit(model, train_loader, val_loader)

    # Save for inference
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path:
        print(f"Best model: {best_path}")
        ckpt = torch.load(best_path, map_location="cpu")
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
                      if k.startswith("model.")}
        os.makedirs("resources", exist_ok=True)
        torch.save(state_dict, "resources/model.pt")
        print("Saved to resources/model.pt")

    # Local evaluation
    print("\n=== Local Evaluation ===")
    y_test_path = os.path.join(args.data_dir, "y_test_reduced.pickle")
    if os.path.exists(y_test_path):
        test_items = build_dataset(
            X_test, None,
            cache_path=os.path.join(args.cache_dir, f"test_dataset_{CACHE_TAG}_nk{N_KERNEL}.pkl"),
            n_workers=args.n_workers,
        )
        test_dataset = CausalEdgeDataset(test_items)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_model = ADIAModel()
        eval_model.load_state_dict(torch.load("resources/model.pt", map_location=device))
        eval_model.to(device)

        predictions = infer_batch_local(eval_model, test_dataset, device=device)
        print(f"Made {len(predictions)} predictions")

        # Compute accuracy against ground truth
        y_test = pd.read_pickle(y_test_path)
        correct = 0
        total = 0
        per_class_correct = np.zeros(N_CLASSES)
        per_class_total = np.zeros(N_CLASSES)

        test_keys = list(X_test.keys())
        for b, key in enumerate(test_keys):
            y_df = y_test[key]
            adj_df = pd.DataFrame(y_df.values, index=list(y_df.columns), columns=list(y_df.columns))
            labels = get_labels(adj_df, _ADJACENCY_LABEL)

            cols = list(X_test[key].columns)
            non_xy = [c for c in cols if c not in ("X", "Y")]
            for v_name in non_xy:
                pred = predictions.get((b, v_name))
                true = labels.get(v_name)
                if pred is not None and true is not None:
                    total += 1
                    if pred == true:
                        correct += 1
                        per_class_correct[true] += 1
                    per_class_total[true] += 1

        overall_acc = correct / total if total > 0 else 0.0
        print(f"\nOverall accuracy: {overall_acc:.4f} ({correct}/{total})")
        print("\nPer-class accuracy:")
        for c in range(N_CLASSES):
            if per_class_total[c] > 0:
                acc = per_class_correct[c] / per_class_total[c]
                print(f"  {CLASS_NAMES[c]:20s}: {acc:.4f} ({int(per_class_correct[c])}/{int(per_class_total[c])})")
    else:
        print(f"No ground truth found at {y_test_path}, skipping evaluation")


if __name__ == "__main__":
    rank = int(os.environ.get("LOCAL_RANK", 0))
    main()