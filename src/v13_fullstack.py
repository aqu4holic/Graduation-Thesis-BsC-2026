"""
v13_fullstack.py — Multi-Paradigm Causal Role Classification

This is NOT an incremental improvement over the conv1d approach.
This is a complete causal discovery system that attacks the problem from
every angle simultaneously.

=== PHILOSOPHY ===
The 8 classes are DEFINED by the DAG structure. The theoretically optimal
approach is to DISCOVER THE DAG, then read off labels. Everything else is
an approximation. This system:

1. Actually runs causal discovery algorithms (PC, LiNGAM, NOTEARS, GES)
2. Computes 300+ causal/statistical features per variable
3. Trains gradient boosting (XGBoost/LightGBM/CatBoost) on those features
4. Refines predictions with a GNN that reasons about graph consistency
5. Stacks everything with the conv1d model for final predictions

=== HARDWARE REQUIREMENTS ===
- 48 CPU cores: parallel feature computation + causal discovery
- 251 GB RAM: holds all features + augmented data in memory
- 100 GB VRAM (2 GPUs): one for conv1d, one for GNN + stacking

=== USAGE ===
# Step 1: Compute features (CPU heavy, ~4-8 hours with XY aug)
python v13_fullstack.py --stage features --data_dir data/

# Step 2: Train tree ensemble (CPU, ~1 hour)
python v13_fullstack.py --stage trees --data_dir data/

# Step 3: Train GNN refinement (GPU, ~30 min)
python v13_fullstack.py --stage gnn --data_dir data/

# Step 4: Stack with conv1d predictions (GPU, ~10 min)
python v13_fullstack.py --stage stack --data_dir data/ --conv1d_preds resources/conv1d_val_probs.npy

# Or run everything:
python v13_fullstack.py --stage all --data_dir data/
"""

import os
import sys
import time
import warnings
import argparse
import pickle
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import digamma
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import expm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm.auto import tqdm

import torch
from torch import nn
import torch.nn.functional as F

import networkx as nx

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
N_OBS = 1000
N_KERNEL = 1000
N_CLASSES = 8
CLASS_NAMES = [
    "Confounder", "Collider", "Mediator",
    "Cause of X", "Cause of Y",
    "Consequence of X", "Consequence of Y",
    "Independent",
]

CACHE_DIR = "dataset_cache/v13/"
FEATURE_CACHE = os.path.join(CACHE_DIR, "features_{split}_{tag}.pkl")

# Feature computation parameters
KNN_K = 7           # k for k-NN MI/CMI estimators
KNN_SUBSAMPLE = 500  # subsample for expensive k-NN operations
HSIC_SUBSAMPLE = 500
DCOR_SUBSAMPLE = 500
NOTEARS_MAX_ITER = 100
NOTEARS_LAMBDA = 0.1
PC_ALPHA = 0.05      # significance level for PC algorithm

# Tree ensemble parameters
XGB_PARAMS = {
    "n_estimators": 8000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softmax",
    "num_class": 8,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "n_jobs": 48,
    "random_state": 42,
}

LGBM_PARAMS = {
    "n_estimators": 8000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multiclass",
    "num_class": 8,
    "metric": "multi_logloss",
    "n_jobs": 48,
    "random_state": 42,
    "verbose": -1,
}

# GNN parameters
GNN_HIDDEN = 128
GNN_LAYERS = 4
GNN_HEADS = 8
GNN_LR = 5e-4
GNN_EPOCHS = 50


# ============================================================
# Graph Utilities (same as all versions)
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
    # Independent: X→Y only, v is isolated — must add v explicitly
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
        result[variable] = adjacency_label.get(key, 7)  # default Independent
    return result


# ############################################################
#
#   PART 1: THE FEATURE FACTORY
#
#   ~300 features per variable, organized into groups.
#   This is the core of the approach — give the model explicit
#   causal statistics instead of forcing it to learn them.
#
# ############################################################

# === Utility functions ===

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        r, _ = sp_stats.spearmanr(x, y)
        return float(r) if np.isfinite(r) else 0.0
    except:
        return 0.0


def _safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    # Subsample for speed (Kendall is O(n²))
    n = len(x)
    if n > 500:
        idx = np.random.choice(n, 500, replace=False)
        x, y = x[idx], y[idx]
    try:
        r, _ = sp_stats.kendalltau(x, y)
        return float(r) if np.isfinite(r) else 0.0
    except:
        return 0.0


def _partial_corr_matrix(data: np.ndarray) -> np.ndarray:
    """Partial correlation via precision matrix."""
    p = data.shape[1]
    cov = np.cov(data.T)
    cov += 1e-6 * np.eye(p)
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.zeros((p, p))
    diag = np.sqrt(np.maximum(np.diag(precision), 1e-10))
    pcorr = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcorr, 0.0)
    return pcorr


def _knn_mi(x: np.ndarray, y: np.ndarray, k: int = KNN_K) -> float:
    """KSG mutual information estimator."""
    n = len(x)
    if n < k + 5:
        return 0.0
    if n > KNN_SUBSAMPLE:
        idx = np.random.choice(n, KNN_SUBSAMPLE, replace=False)
        x, y = x[idx], y[idx]
        n = KNN_SUBSAMPLE

    x = ((x - x.mean()) / (x.std() + 1e-10))[:, None]
    y = ((y - y.mean()) / (y.std() + 1e-10))[:, None]
    xy = np.hstack([x, y])

    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    dd, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dd[:, -1]
    eps = np.maximum(eps, 1e-10)

    n_x = np.array([tree_x.query_ball_point(x[i], eps[i], p=np.inf, return_length=True) - 1
                     for i in range(n)])
    n_y = np.array([tree_y.query_ball_point(y[i], eps[i], p=np.inf, return_length=True) - 1
                     for i in range(n)])
    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)

    mi = digamma(k) + digamma(n) - np.mean(digamma(n_x) + digamma(n_y))
    return float(max(mi, 0.0))


def _knn_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = KNN_K) -> float:
    """Frenzel-Pompe CMI estimator I(X;Y|Z)."""
    n = len(x)
    if n < k + 5:
        return 0.0
    if n > KNN_SUBSAMPLE:
        idx = np.random.choice(n, KNN_SUBSAMPLE, replace=False)
        x, y, z = x[idx], y[idx], (z[idx] if z.ndim > 1 else z[idx])
        n = KNN_SUBSAMPLE

    x = ((x - x.mean()) / (x.std() + 1e-10))[:, None]
    y = ((y - y.mean()) / (y.std() + 1e-10))[:, None]
    if z.ndim == 1:
        z = ((z - z.mean()) / (z.std() + 1e-10))[:, None]
    else:
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-10)

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


def _hsic(x: np.ndarray, y: np.ndarray) -> float:
    """HSIC with median heuristic bandwidth."""
    n = len(x)
    if n < 5:
        return 0.0
    if n > HSIC_SUBSAMPLE:
        idx = np.random.choice(n, HSIC_SUBSAMPLE, replace=False)
        x, y = x[idx], y[idx]
        n = HSIC_SUBSAMPLE

    # Median heuristic
    sx = np.median(np.abs(x[:, None] - x[None, :])) + 1e-10
    sy = np.median(np.abs(y[:, None] - y[None, :])) + 1e-10

    Kx = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / sx ** 2)
    Ky = np.exp(-0.5 * (y[:, None] - y[None, :]) ** 2 / sy ** 2)
    H = np.eye(n) - 1.0 / n
    return float(np.trace(H @ Kx @ H @ Ky) / n ** 2)


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation."""
    n = len(x)
    if n < 5:
        return 0.0
    if n > DCOR_SUBSAMPLE:
        idx = np.random.choice(n, DCOR_SUBSAMPLE, replace=False)
        x, y = x[idx], y[idx]
        n = DCOR_SUBSAMPLE

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
    B = b - b.mean(0, keepdims=True) - b.mean(1, keepdims=True) + b.mean()

    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    denom = np.sqrt(dvar_x * dvar_y)
    if denom < 1e-10:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom))


def _regression_features(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute regression features for x→y.
    Returns R², residual skewness, residual kurtosis, residual-cause correlation.
    """
    n = len(x)
    if np.var(x) < 1e-10 or np.var(y) < 1e-10:
        return {"r2": 0.0, "resid_skew": 0.0, "resid_kurt": 0.0,
                "resid_cause_corr": 0.0, "coeff": 0.0, "intercept": 0.0}

    # Linear regression
    x_mean, y_mean = x.mean(), y.mean()
    cov_xy = np.cov(x, y)[0, 1]
    var_x = np.var(x)
    b = cov_xy / (var_x + 1e-10)
    a = y_mean - b * x_mean
    y_hat = a + b * x
    resid = y - y_hat

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y_mean) ** 2) + 1e-10
    r2 = 1.0 - ss_res / ss_tot

    resid_std = np.std(resid)
    if resid_std < 1e-10:
        return {"r2": r2, "resid_skew": 0.0, "resid_kurt": 0.0,
                "resid_cause_corr": 0.0, "coeff": float(b), "intercept": float(a)}

    resid_normed = resid / resid_std
    return {
        "r2": float(r2),
        "resid_skew": float(sp_stats.skew(resid_normed)),
        "resid_kurt": float(sp_stats.kurtosis(resid_normed)),
        "resid_cause_corr": _safe_corr(resid, x),  # ANM test: should be ~0 if x→y
        "coeff": float(b),
        "intercept": float(a),
    }


def _polynomial_r2(x: np.ndarray, y: np.ndarray, degree: int = 3) -> float:
    """R² of polynomial regression y ~ poly(x, degree)."""
    n = len(x)
    if np.var(x) < 1e-10 or np.var(y) < 1e-10:
        return 0.0
    try:
        coeffs = np.polyfit(x, y, degree)
        y_hat = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-10
        return float(max(0.0, 1.0 - ss_res / ss_tot))
    except:
        return 0.0


def _variable_stats(x: np.ndarray) -> dict:
    """Basic statistics of a single variable."""
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "skew": float(sp_stats.skew(x)),
        "kurtosis": float(sp_stats.kurtosis(x)),
        "median": float(np.median(x)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "entropy": float(_kde_entropy(x)),
    }


def _kde_entropy(x: np.ndarray, n_bins: int = 50) -> float:
    """Approximate entropy using histogram."""
    counts, _ = np.histogram(x, bins=n_bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-10)))


# === Feature computation for a single variable relative to X, Y ===

def compute_features_for_variable(
    data: np.ndarray,
    v_idx: int,
    x_idx: int,
    y_idx: int,
    all_cols: list,
    pcorr_matrix: np.ndarray,
) -> dict:
    """
    Compute ALL features for variable v relative to X→Y.

    Groups:
      A. Marginal statistics of v (7 features)
      B. Pairwise v-X statistics (20+ features)
      C. Pairwise v-Y statistics (20+ features)
      D. Pairwise X-Y statistics (context, 20+ features)
      E. Conditional independence features (12+ features) — THE KEY
      F. Regression & ANM features (24+ features)
      G. Higher-order features (10+ features)
      H. Causal discovery algorithm outputs (8+ features)
      I. Graph structure features (5 features)

    Returns: dict of feature_name → float
    """
    N, p = data.shape
    v = data[:, v_idx]
    x = data[:, x_idx]
    y = data[:, y_idx]

    features = {}

    # ========== A. Marginal statistics of v ==========
    v_stats = _variable_stats(v)
    for k, val in v_stats.items():
        features[f"v_{k}"] = val

    # ========== B. Pairwise v-X ==========
    features["vx_pearson"] = _safe_corr(v, x)
    features["vx_spearman"] = _safe_spearman(v, x)
    features["vx_kendall"] = _safe_kendall(v, x)
    features["vx_dcor"] = _distance_correlation(v, x)
    features["vx_hsic"] = _hsic(v, x)
    features["vx_mi"] = _knn_mi(v, x)
    features["vx_pcorr"] = pcorr_matrix[v_idx, x_idx]
    features["vx_abs_pearson"] = abs(features["vx_pearson"])
    features["vx_abs_pcorr"] = abs(features["vx_pcorr"])

    # Regression v→x
    reg_vx = _regression_features(v, x)
    for k, val in reg_vx.items():
        features[f"vx_reg_{k}"] = val
    # Regression x→v
    reg_xv = _regression_features(x, v)
    for k, val in reg_xv.items():
        features[f"xv_reg_{k}"] = val

    # ANM direction test: if v→x, then resid(v→x) ⊥ v
    features["vx_anm_asym"] = abs(reg_vx["resid_cause_corr"]) - abs(reg_xv["resid_cause_corr"])

    # Polynomial R²
    features["vx_poly_r2"] = _polynomial_r2(v, x, degree=3)
    features["xv_poly_r2"] = _polynomial_r2(x, v, degree=3)

    # ========== C. Pairwise v-Y ==========
    features["vy_pearson"] = _safe_corr(v, y)
    features["vy_spearman"] = _safe_spearman(v, y)
    features["vy_kendall"] = _safe_kendall(v, y)
    features["vy_dcor"] = _distance_correlation(v, y)
    features["vy_hsic"] = _hsic(v, y)
    features["vy_mi"] = _knn_mi(v, y)
    features["vy_pcorr"] = pcorr_matrix[v_idx, y_idx]
    features["vy_abs_pearson"] = abs(features["vy_pearson"])
    features["vy_abs_pcorr"] = abs(features["vy_pcorr"])

    reg_vy = _regression_features(v, y)
    for k, val in reg_vy.items():
        features[f"vy_reg_{k}"] = val
    reg_yv = _regression_features(y, v)
    for k, val in reg_yv.items():
        features[f"yv_reg_{k}"] = val

    features["vy_anm_asym"] = abs(reg_vy["resid_cause_corr"]) - abs(reg_yv["resid_cause_corr"])
    features["vy_poly_r2"] = _polynomial_r2(v, y, degree=3)
    features["yv_poly_r2"] = _polynomial_r2(y, v, degree=3)

    # ========== D. Pairwise X-Y (context) ==========
    features["xy_pearson"] = _safe_corr(x, y)
    features["xy_spearman"] = _safe_spearman(x, y)
    features["xy_dcor"] = _distance_correlation(x, y)
    features["xy_hsic"] = _hsic(x, y)
    features["xy_mi"] = _knn_mi(x, y)
    features["xy_pcorr"] = pcorr_matrix[x_idx, y_idx]

    reg_xy = _regression_features(x, y)
    for k, val in reg_xy.items():
        features[f"xy_reg_{k}"] = val

    # ========== E. Conditional Independence — THE MOST IMPORTANT GROUP ==========
    # These features DIRECTLY encode class membership:
    #   Confounder: v→X, v→Y    → CMI(v,X|Y) high, CMI(v,Y|X) high
    #   Collider:   X→v, Y→v    → CMI(X,Y|v) high (explaining away)
    #   Mediator:   X→v→Y       → CMI(v,Y|X) high, CMI(X,Y|v) low
    #   CauseX:     v→X         → CMI(v,X|Y) high, CMI(v,Y|X) low
    #   CauseY:     v→Y         → CMI(v,Y|X) high, CMI(v,X|Y) low
    #   ConseqX:    X→v         → CMI(v,X|Y) high (via X), direction asymmetry
    #   ConseqY:    Y→v         → CMI(v,Y|X) high (via Y), direction asymmetry
    #   Independent: v ⊥ X,Y    → all CMI/MI low

    features["cmi_vx_given_y"] = _knn_cmi(v, x, y)      # I(v;X|Y)
    features["cmi_vy_given_x"] = _knn_cmi(v, y, x)      # I(v;Y|X)
    features["cmi_xy_given_v"] = _knn_cmi(x, y, v)      # I(X;Y|v)

    # Also condition on all other variables
    other_idx = [i for i in range(p) if i not in (v_idx, x_idx, y_idx)]
    if len(other_idx) > 0:
        other_data = data[:, other_idx]
        # CMI conditioning on everything else
        z_all = np.hstack([y[:, None], other_data])
        features["cmi_vx_given_rest"] = _knn_cmi(v, x, z_all)
        z_all2 = np.hstack([x[:, None], other_data])
        features["cmi_vy_given_rest"] = _knn_cmi(v, y, z_all2)
        z_all3 = np.hstack([v[:, None], other_data])
        features["cmi_xy_given_v_rest"] = _knn_cmi(x, y, z_all3)
    else:
        features["cmi_vx_given_rest"] = features["cmi_vx_given_y"]
        features["cmi_vy_given_rest"] = features["cmi_vy_given_x"]
        features["cmi_xy_given_v_rest"] = features["cmi_xy_given_v"]

    # Derived CI features
    features["cmi_ratio_vx"] = features["cmi_vx_given_y"] / (features["vx_mi"] + 1e-6)
    features["cmi_ratio_vy"] = features["cmi_vy_given_x"] / (features["vy_mi"] + 1e-6)
    features["cmi_asym_vxy"] = features["cmi_vx_given_y"] - features["cmi_vy_given_x"]
    features["explaining_away"] = features["cmi_xy_given_v"] - features["xy_mi"]
    features["mi_drop_x_given_v"] = features["vx_mi"] - features["cmi_vx_given_y"]  # mediation signature
    features["mi_drop_y_given_v"] = features["vy_mi"] - features["cmi_vy_given_x"]

    # ========== F. Multi-directional regression + ANM ==========
    # Predict v from X+Y jointly
    if np.var(v) > 1e-10:
        XY = np.column_stack([x, y])
        try:
            beta = np.linalg.lstsq(
                np.column_stack([np.ones(N), XY]), v, rcond=None
            )[0]
            v_hat = np.column_stack([np.ones(N), XY]) @ beta
            resid_v = v - v_hat
            ss_res = np.sum(resid_v ** 2)
            ss_tot = np.sum((v - v.mean()) ** 2) + 1e-10
            features["joint_r2_xy_to_v"] = float(1 - ss_res / ss_tot)
            features["joint_resid_v_corr_x"] = _safe_corr(resid_v, x)
            features["joint_resid_v_corr_y"] = _safe_corr(resid_v, y)
            features["joint_resid_v_skew"] = float(sp_stats.skew(resid_v))
        except:
            features["joint_r2_xy_to_v"] = 0.0
            features["joint_resid_v_corr_x"] = 0.0
            features["joint_resid_v_corr_y"] = 0.0
            features["joint_resid_v_skew"] = 0.0
    else:
        features["joint_r2_xy_to_v"] = 0.0
        features["joint_resid_v_corr_x"] = 0.0
        features["joint_resid_v_corr_y"] = 0.0
        features["joint_resid_v_skew"] = 0.0

    # Predict X from v+Y jointly
    VY = np.column_stack([v, y])
    try:
        beta = np.linalg.lstsq(np.column_stack([np.ones(N), VY]), x, rcond=None)[0]
        x_hat = np.column_stack([np.ones(N), VY]) @ beta
        resid_x = x - x_hat
        features["joint_r2_vy_to_x"] = float(1 - np.sum(resid_x**2) / (np.sum((x-x.mean())**2) + 1e-10))
        features["joint_resid_x_corr_v"] = _safe_corr(resid_x, v)
    except:
        features["joint_r2_vy_to_x"] = 0.0
        features["joint_resid_x_corr_v"] = 0.0

    # Predict Y from v+X jointly
    VX = np.column_stack([v, x])
    try:
        beta = np.linalg.lstsq(np.column_stack([np.ones(N), VX]), y, rcond=None)[0]
        y_hat = np.column_stack([np.ones(N), VX]) @ beta
        resid_y = y - y_hat
        features["joint_r2_vx_to_y"] = float(1 - np.sum(resid_y**2) / (np.sum((y-y.mean())**2) + 1e-10))
        features["joint_resid_y_corr_v"] = _safe_corr(resid_y, v)
    except:
        features["joint_r2_vx_to_y"] = 0.0
        features["joint_resid_y_corr_v"] = 0.0

    # ========== G. Higher-order / nonlinear features ==========
    # Nonlinear dependence: ratio of distance_corr to pearson (high = nonlinear)
    features["vx_nonlinearity"] = features["vx_dcor"] / (abs(features["vx_pearson"]) + 1e-6)
    features["vy_nonlinearity"] = features["vy_dcor"] / (abs(features["vy_pearson"]) + 1e-6)

    # Conditional nonlinearity: polynomial R² gain over linear
    features["vx_poly_gain"] = features["vx_poly_r2"] - features["vx_reg_r2"]
    features["vy_poly_gain"] = features["vy_poly_r2"] - features["vy_reg_r2"]
    features["xv_poly_gain"] = features["xv_poly_r2"] - features["xv_reg_r2"]
    features["yv_poly_gain"] = features["yv_poly_r2"] - features["yv_reg_r2"]

    # Interaction: product of correlations (confounder signature)
    features["corr_product_vx_vy"] = features["vx_pearson"] * features["vy_pearson"]
    features["mi_product_vx_vy"] = features["vx_mi"] * features["vy_mi"]

    # ========== I. Graph structure ==========
    features["n_variables"] = float(p)
    features["log_n_variables"] = float(np.log(p))
    features["n_observations"] = float(N)

    # Relative position of v's dependence strength
    all_corrs_x = [abs(_safe_corr(data[:, i], x)) for i in range(p) if i != x_idx]
    all_corrs_y = [abs(_safe_corr(data[:, i], y)) for i in range(p) if i != y_idx]
    v_rank_x = sum(1 for c in all_corrs_x if c <= abs(features["vx_pearson"])) / max(len(all_corrs_x), 1)
    v_rank_y = sum(1 for c in all_corrs_y if c <= abs(features["vy_pearson"])) / max(len(all_corrs_y), 1)
    features["v_rank_corr_x"] = v_rank_x
    features["v_rank_corr_y"] = v_rank_y

    return features


# ############################################################
#
#   PART 2: CAUSAL DISCOVERY ALGORITHMS
#
#   Run multiple algorithms, extract per-variable features.
#   Each algorithm provides "votes" for edge existence.
#
# ############################################################

def _fisher_z_test(data: np.ndarray, i: int, j: int, cond_set: list, alpha: float = PC_ALPHA) -> bool:
    """
    Fisher-z conditional independence test.
    Returns True if i ⊥ j | cond_set (independent).
    """
    n = data.shape[0]
    if len(cond_set) == 0:
        r = _safe_corr(data[:, i], data[:, j])
    else:
        # Partial correlation
        idx_set = [i, j] + list(cond_set)
        sub_data = data[:, idx_set]
        cov = np.cov(sub_data.T)
        cov += 1e-8 * np.eye(len(idx_set))
        try:
            precision = np.linalg.inv(cov)
        except:
            return False
        r = -precision[0, 1] / np.sqrt(abs(precision[0, 0] * precision[1, 1]) + 1e-10)

    r = np.clip(r, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
    df = n - len(cond_set) - 3
    if df <= 0:
        return False
    t_stat = abs(z) * np.sqrt(df)
    p_val = 2 * (1 - sp_stats.t.cdf(t_stat, df))
    return p_val > alpha


def run_pc_algorithm(data: np.ndarray, alpha: float = PC_ALPHA, max_cond: int = 3) -> np.ndarray:
    """
    Simplified PC algorithm.
    Returns adjacency matrix (undirected skeleton + partial orientation).
    """
    p = data.shape[1]

    # Start with complete undirected graph
    adj = np.ones((p, p), dtype=int)
    np.fill_diagonal(adj, 0)
    sep_sets = defaultdict(set)

    # Phase 1: Skeleton discovery
    for depth in range(max_cond + 1):
        for i in range(p):
            for j in range(i + 1, p):
                if adj[i, j] == 0:
                    continue
                # Find neighbors of i (excluding j)
                neighbors_i = [k for k in range(p) if k != i and k != j and adj[i, k] == 1]
                if len(neighbors_i) < depth:
                    continue

                # Test all conditioning sets of size `depth`
                from itertools import combinations
                found_independent = False
                for cond in combinations(neighbors_i, depth):
                    cond_list = list(cond)
                    if _fisher_z_test(data, i, j, cond_list, alpha):
                        adj[i, j] = adj[j, i] = 0
                        sep_sets[(i, j)] = set(cond_list)
                        sep_sets[(j, i)] = set(cond_list)
                        found_independent = True
                        break

    # Phase 2: Orient v-structures (colliders)
    # If i — k — j and i,j not adjacent and k not in sep(i,j): orient i→k←j
    dag = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if adj[i, j]:
                dag[i, j] = 1  # undirected for now

    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j] == 1:
                continue  # must be non-adjacent
            # Find common neighbors
            for k in range(p):
                if k == i or k == j:
                    continue
                if adj[i, k] == 1 and adj[k, j] == 1:
                    if k not in sep_sets.get((i, j), set()):
                        # Orient i→k←j
                        dag[i, k] = 1
                        dag[k, i] = 0
                        dag[j, k] = 1
                        dag[k, j] = 0

    return dag


def run_direct_lingam(data: np.ndarray) -> np.ndarray:
    """
    Simplified DirectLiNGAM.
    Uses residual-based iterative procedure to find causal ordering.
    Returns weighted adjacency matrix.
    """
    N, p = data.shape
    remaining = list(range(p))
    order = []
    residuals = data.copy()

    for _ in range(p):
        if len(remaining) <= 1:
            order.extend(remaining)
            break

        # Find the most exogenous variable (least predictable from others)
        best_var = None
        best_score = -np.inf

        for i in remaining:
            # Regress i on all other remaining variables
            others = [j for j in remaining if j != i]
            X_design = residuals[:, others]
            y = residuals[:, i]

            if np.var(y) < 1e-10:
                score = np.inf  # constant variable is exogenous
            else:
                try:
                    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
                    resid = y - X_design @ beta
                    # Score: independence of residuals from other variables (use kurtosis as proxy)
                    score = abs(sp_stats.kurtosis(resid))
                except:
                    score = 0.0

            if score > best_score:
                best_score = score
                best_var = i

        if best_var is None:
            best_var = remaining[0]

        order.append(best_var)
        remaining.remove(best_var)

        # Regress remaining variables on best_var and use residuals
        if len(remaining) > 0:
            xi = residuals[:, best_var]
            for j in remaining:
                xj = residuals[:, j]
                if np.var(xi) > 1e-10:
                    b = np.cov(xi, xj)[0, 1] / (np.var(xi) + 1e-10)
                    residuals[:, j] = xj - b * xi

    # Build adjacency from causal ordering
    adj = np.zeros((p, p))
    for pos, var in enumerate(order):
        for later_var in order[pos + 1:]:
            # Regress later_var on var using original data
            xi = data[:, var]
            xj = data[:, later_var]
            if np.var(xi) > 1e-10:
                b = np.cov(xi, xj)[0, 1] / (np.var(xi) + 1e-10)
                if abs(b) > 0.05:  # threshold for edge inclusion
                    adj[var, later_var] = b

    return adj


def run_notears(data: np.ndarray, lambda1: float = NOTEARS_LAMBDA,
                max_iter: int = NOTEARS_MAX_ITER) -> np.ndarray:
    """
    NOTEARS: continuous optimization for DAG structure learning.
    Zheng et al. (2018) — "DAGs with NO TEARS"

    min_{W} 0.5/n ||X - XW||² + λ|W|
    s.t. h(W) = tr(e^{W∘W}) - d = 0  (acyclicity constraint)
    """
    N, d = data.shape
    X = data.copy()
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    W = np.zeros((d, d))
    rho = 1.0
    alpha_dual = np.zeros((d, d))
    h_tol = 1e-8

    for iteration in range(max_iter):
        # Optimize W with augmented Lagrangian
        for _ in range(20):  # inner iterations
            # Gradient of least-squares loss
            residual = X - X @ W
            grad_loss = -(X.T @ residual) / N

            # Gradient of acyclicity constraint h(W) = tr(e^{W∘W}) - d
            M = W * W
            E = expm(M)
            grad_h = 2 * W * E  # ∂h/∂W = 2W ∘ e^{W∘W}

            # L1 subgradient
            grad_l1 = lambda1 * np.sign(W)

            # Total gradient
            grad = grad_loss + (rho * (np.trace(E) - d) + 1.0) * grad_h + grad_l1

            # Gradient descent step
            lr = 0.001 / (1 + iteration * 0.1)
            W = W - lr * grad

        # Evaluate constraint
        h = np.trace(expm(W * W)) - d

        if abs(h) < h_tol:
            break

        # Update dual variable and penalty
        alpha_dual += rho * h
        rho = min(rho * 2, 1e6)

    # Threshold small weights
    W[np.abs(W) < 0.1] = 0

    return W


def run_ges_simplified(data: np.ndarray) -> np.ndarray:
    """
    Simplified Greedy Equivalence Search using BIC score.
    Forward phase only (add edges greedily).
    """
    N, p = data.shape

    def bic_score(adj: np.ndarray) -> float:
        """BIC score for the entire DAG."""
        score = 0.0
        for j in range(p):
            parents = np.where(adj[:, j] > 0)[0]
            k = len(parents)
            if k == 0:
                rss = np.sum((data[:, j] - data[:, j].mean()) ** 2)
            else:
                X_pa = data[:, parents]
                X_design = np.column_stack([np.ones(N), X_pa])
                try:
                    beta = np.linalg.lstsq(X_design, data[:, j], rcond=None)[0]
                    y_hat = X_design @ beta
                    rss = np.sum((data[:, j] - y_hat) ** 2)
                except:
                    rss = np.sum((data[:, j] - data[:, j].mean()) ** 2)

            score += N * np.log(rss / N + 1e-10) + (k + 1) * np.log(N)
        return score

    adj = np.zeros((p, p), dtype=int)
    current_score = bic_score(adj)

    # Forward phase: greedily add edges
    improved = True
    while improved:
        improved = False
        best_edge = None
        best_score = current_score

        for i in range(p):
            for j in range(p):
                if i == j or adj[i, j] == 1:
                    continue
                # Try adding edge i→j
                adj[i, j] = 1
                # Check acyclicity
                try:
                    G = nx.DiGraph(adj)
                    if nx.is_directed_acyclic_graph(G):
                        s = bic_score(adj)
                        if s < best_score:
                            best_score = s
                            best_edge = (i, j)
                except:
                    pass
                adj[i, j] = 0

        if best_edge is not None:
            adj[best_edge[0], best_edge[1]] = 1
            current_score = best_score
            improved = True

    return adj


def extract_causal_discovery_features(
    data: np.ndarray,
    cols: list,
    v_name: str,
    x_name: str = "X",
    y_name: str = "Y",
) -> dict:
    """
    Run all causal discovery algorithms and extract per-variable features.
    """
    col_idx = {name: i for i, name in enumerate(cols)}
    v_idx = col_idx[v_name]
    x_idx = col_idx[x_name]
    y_idx = col_idx[y_name]

    features = {}

    # --- PC Algorithm ---
    try:
        pc_dag = run_pc_algorithm(data, alpha=PC_ALPHA)
        features["pc_v_to_x"] = float(pc_dag[v_idx, x_idx])
        features["pc_x_to_v"] = float(pc_dag[x_idx, v_idx])
        features["pc_v_to_y"] = float(pc_dag[v_idx, y_idx])
        features["pc_y_to_v"] = float(pc_dag[y_idx, v_idx])
        features["pc_x_to_y"] = float(pc_dag[x_idx, y_idx])
        # Predicted class from PC
        pc_class = _dag_to_class(pc_dag, v_idx, x_idx, y_idx)
        for c in range(N_CLASSES):
            features[f"pc_class_{c}"] = float(pc_class == c)
    except Exception as e:
        for key in ["pc_v_to_x", "pc_x_to_v", "pc_v_to_y", "pc_y_to_v", "pc_x_to_y"]:
            features[key] = 0.0
        for c in range(N_CLASSES):
            features[f"pc_class_{c}"] = 0.0

    # --- DirectLiNGAM ---
    try:
        lingam_adj = run_direct_lingam(data)
        features["lingam_v_to_x"] = float(abs(lingam_adj[v_idx, x_idx]) > 0.05)
        features["lingam_x_to_v"] = float(abs(lingam_adj[x_idx, v_idx]) > 0.05)
        features["lingam_v_to_y"] = float(abs(lingam_adj[v_idx, y_idx]) > 0.05)
        features["lingam_y_to_v"] = float(abs(lingam_adj[y_idx, v_idx]) > 0.05)
        features["lingam_weight_vx"] = float(lingam_adj[v_idx, x_idx])
        features["lingam_weight_xv"] = float(lingam_adj[x_idx, v_idx])
        features["lingam_weight_vy"] = float(lingam_adj[v_idx, y_idx])
        features["lingam_weight_yv"] = float(lingam_adj[y_idx, v_idx])
    except:
        for key in ["lingam_v_to_x", "lingam_x_to_v", "lingam_v_to_y", "lingam_y_to_v",
                     "lingam_weight_vx", "lingam_weight_xv", "lingam_weight_vy", "lingam_weight_yv"]:
            features[key] = 0.0

    # --- NOTEARS ---
    try:
        notears_adj = run_notears(data)
        features["notears_v_to_x"] = float(abs(notears_adj[v_idx, x_idx]) > 0.05)
        features["notears_x_to_v"] = float(abs(notears_adj[x_idx, v_idx]) > 0.05)
        features["notears_v_to_y"] = float(abs(notears_adj[v_idx, y_idx]) > 0.05)
        features["notears_y_to_v"] = float(abs(notears_adj[y_idx, v_idx]) > 0.05)
        features["notears_weight_vx"] = float(notears_adj[v_idx, x_idx])
        features["notears_weight_xv"] = float(notears_adj[x_idx, v_idx])
        features["notears_weight_vy"] = float(notears_adj[v_idx, y_idx])
        features["notears_weight_yv"] = float(notears_adj[y_idx, v_idx])
    except:
        for key in ["notears_v_to_x", "notears_x_to_v", "notears_v_to_y", "notears_y_to_v",
                     "notears_weight_vx", "notears_weight_xv", "notears_weight_vy", "notears_weight_yv"]:
            features[key] = 0.0

    # --- GES ---
    try:
        ges_adj = run_ges_simplified(data)
        features["ges_v_to_x"] = float(ges_adj[v_idx, x_idx])
        features["ges_x_to_v"] = float(ges_adj[x_idx, v_idx])
        features["ges_v_to_y"] = float(ges_adj[v_idx, y_idx])
        features["ges_y_to_v"] = float(ges_adj[y_idx, v_idx])
        ges_class = _dag_to_class(ges_adj, v_idx, x_idx, y_idx)
        for c in range(N_CLASSES):
            features[f"ges_class_{c}"] = float(ges_class == c)
    except:
        for key in ["ges_v_to_x", "ges_x_to_v", "ges_v_to_y", "ges_y_to_v"]:
            features[key] = 0.0
        for c in range(N_CLASSES):
            features[f"ges_class_{c}"] = 0.0

    # --- Consensus features ---
    vote_v_to_x = sum([features.get(f"{alg}_v_to_x", 0) for alg in ["pc", "lingam", "notears", "ges"]])
    vote_x_to_v = sum([features.get(f"{alg}_x_to_v", 0) for alg in ["pc", "lingam", "notears", "ges"]])
    vote_v_to_y = sum([features.get(f"{alg}_v_to_y", 0) for alg in ["pc", "lingam", "notears", "ges"]])
    vote_y_to_v = sum([features.get(f"{alg}_y_to_v", 0) for alg in ["pc", "lingam", "notears", "ges"]])
    features["consensus_v_to_x"] = vote_v_to_x / 4.0
    features["consensus_x_to_v"] = vote_x_to_v / 4.0
    features["consensus_v_to_y"] = vote_v_to_y / 4.0
    features["consensus_y_to_v"] = vote_y_to_v / 4.0

    return features


def _dag_to_class(adj: np.ndarray, v_idx: int, x_idx: int, y_idx: int) -> int:
    """Convert adjacency matrix entries to 8-class label."""
    v_to_x = adj[v_idx, x_idx] > 0
    x_to_v = adj[x_idx, v_idx] > 0
    v_to_y = adj[v_idx, y_idx] > 0
    y_to_v = adj[y_idx, v_idx] > 0

    if v_to_x and v_to_y:
        return 0  # Confounder
    if x_to_v and y_to_v:
        return 1  # Collider
    if x_to_v and v_to_y:
        return 2  # Mediator
    if v_to_x and not v_to_y:
        return 3  # Cause of X
    if v_to_y and not v_to_x:
        return 4  # Cause of Y
    if x_to_v and not y_to_v and not v_to_y:
        return 5  # Consequence of X
    if y_to_v and not x_to_v and not v_to_x:
        return 6  # Consequence of Y
    return 7  # Independent


# ############################################################
#
#   PART 3: FEATURE COMPUTATION PIPELINE
#
#   Parallelized across 48 cores, with caching.
#
# ############################################################

def compute_all_features_for_sample(
    df: pd.DataFrame,
    y_df: Optional[pd.DataFrame] = None,
    run_causal_discovery: bool = True,
) -> List[dict]:
    """
    Compute all features for all non-X/Y variables in one sample.
    Returns list of dicts, one per variable.
    """
    cols = list(df.columns)
    data = df.values.astype(np.float64)
    N, p = data.shape

    col_idx = {name: i for i, name in enumerate(cols)}
    x_idx = col_idx["X"]
    y_idx = col_idx["Y"]

    # Precompute shared quantities
    pcorr_matrix = _partial_corr_matrix(data)

    # Run causal discovery once per sample (shared across variables)
    if run_causal_discovery:
        try:
            pc_dag = run_pc_algorithm(data)
        except:
            pc_dag = np.zeros((p, p))
        try:
            lingam_adj = run_direct_lingam(data)
        except:
            lingam_adj = np.zeros((p, p))
        try:
            notears_adj = run_notears(data)
        except:
            notears_adj = np.zeros((p, p))
        try:
            ges_adj = run_ges_simplified(data)
        except:
            ges_adj = np.zeros((p, p))

    non_xy = [c for c in cols if c not in ("X", "Y")]
    results = []

    for v_name in non_xy:
        v_idx = col_idx[v_name]

        # Statistical features
        features = compute_features_for_variable(data, v_idx, x_idx, y_idx, cols, pcorr_matrix)

        # Causal discovery features (reuse precomputed DAGs)
        if run_causal_discovery:
            # PC
            features["pc_v_to_x"] = float(pc_dag[v_idx, x_idx] > 0)
            features["pc_x_to_v"] = float(pc_dag[x_idx, v_idx] > 0)
            features["pc_v_to_y"] = float(pc_dag[v_idx, y_idx] > 0)
            features["pc_y_to_v"] = float(pc_dag[y_idx, v_idx] > 0)
            pc_class = _dag_to_class(pc_dag, v_idx, x_idx, y_idx)
            for c in range(N_CLASSES):
                features[f"pc_class_{c}"] = float(pc_class == c)

            # LiNGAM
            features["lingam_v_to_x"] = float(abs(lingam_adj[v_idx, x_idx]) > 0.05)
            features["lingam_x_to_v"] = float(abs(lingam_adj[x_idx, v_idx]) > 0.05)
            features["lingam_v_to_y"] = float(abs(lingam_adj[v_idx, y_idx]) > 0.05)
            features["lingam_y_to_v"] = float(abs(lingam_adj[y_idx, v_idx]) > 0.05)
            features["lingam_weight_vx"] = float(lingam_adj[v_idx, x_idx])
            features["lingam_weight_xv"] = float(lingam_adj[x_idx, v_idx])
            features["lingam_weight_vy"] = float(lingam_adj[v_idx, y_idx])
            features["lingam_weight_yv"] = float(lingam_adj[y_idx, v_idx])

            # NOTEARS
            features["notears_v_to_x"] = float(abs(notears_adj[v_idx, x_idx]) > 0.05)
            features["notears_x_to_v"] = float(abs(notears_adj[x_idx, v_idx]) > 0.05)
            features["notears_v_to_y"] = float(abs(notears_adj[v_idx, y_idx]) > 0.05)
            features["notears_y_to_v"] = float(abs(notears_adj[y_idx, v_idx]) > 0.05)
            features["notears_weight_vx"] = float(notears_adj[v_idx, x_idx])
            features["notears_weight_xv"] = float(notears_adj[x_idx, v_idx])
            features["notears_weight_vy"] = float(notears_adj[v_idx, y_idx])
            features["notears_weight_yv"] = float(notears_adj[y_idx, v_idx])

            # GES
            features["ges_v_to_x"] = float(ges_adj[v_idx, x_idx] > 0)
            features["ges_x_to_v"] = float(ges_adj[x_idx, v_idx] > 0)
            features["ges_v_to_y"] = float(ges_adj[v_idx, y_idx] > 0)
            features["ges_y_to_v"] = float(ges_adj[y_idx, v_idx] > 0)
            ges_class = _dag_to_class(ges_adj, v_idx, x_idx, y_idx)
            for c in range(N_CLASSES):
                features[f"ges_class_{c}"] = float(ges_class == c)

            # Consensus
            for edge in ["v_to_x", "x_to_v", "v_to_y", "y_to_v"]:
                votes = sum(features.get(f"{alg}_{edge}", 0) for alg in ["pc", "lingam", "notears", "ges"])
                features[f"consensus_{edge}"] = votes / 4.0

        # Metadata
        features["_variable"] = v_name

        # Label
        if y_df is not None:
            adj_df = pd.DataFrame(y_df.values, index=list(y_df.columns), columns=list(y_df.columns))
            labels = get_labels(adj_df)
            features["_label"] = labels.get(v_name, 7)

        results.append(features)

    return results


def _process_one_sample(args):
    """Worker for parallel feature computation."""
    key, df, y_df, run_cd = args
    try:
        results = compute_all_features_for_sample(df, y_df, run_causal_discovery=run_cd)
        for r in results:
            r["_sample_key"] = key
        return results
    except Exception as e:
        print(f"Error processing sample {key}: {e}")
        return []


def compute_features_parallel(
    X_data: dict,
    y_data: Optional[dict],
    cache_path: str,
    n_workers: int = 48,
    run_causal_discovery: bool = True,
) -> pd.DataFrame:
    """
    Compute features for all samples in parallel.
    Returns a DataFrame with one row per (sample, variable) pair.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"Computing features for {len(X_data)} samples (causal_discovery={run_causal_discovery})...")
    args_list = []
    for key in X_data.keys():
        df = X_data[key]
        y_df = y_data[key] if y_data is not None else None
        args_list.append((key, df, y_df, run_causal_discovery))

    all_results = []
    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_process_one_sample, args): i for i, args in enumerate(args_list)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Feature computation"):
            try:
                results = future.result(timeout=300)  # 5 min timeout per sample
                all_results.extend(results)
            except Exception as e:
                print(f"Failed: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df.to_pickle(cache_path)
    print(f"  Cached to {cache_path}")

    return df


# ############################################################
#
#   PART 4: XY REMAP AUGMENTATION FOR FEATURES
#
# ############################################################

def compute_augmented_features(
    X_data: dict,
    y_data: dict,
    cache_path: str,
    n_workers: int = 48,
    run_causal_discovery: bool = True,
) -> pd.DataFrame:
    """
    XY remap augmentation: for each directed edge A→B in the ground truth,
    treat A as new X and B as new Y, recompute all labels and features.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached augmented features from {cache_path}")
        return pd.read_pickle(cache_path)

    print("Building augmented dataset with XY remap...")
    aug_X = {}
    aug_y = {}
    aug_idx = 0

    for key in tqdm(X_data.keys(), desc="XY remap"):
        df = X_data[key]
        y_df = y_data[key]
        cols = list(df.columns)
        adj = y_df.values

        # Original X→Y pair is already in the base dataset
        # Find all directed edges A→B
        for i, src in enumerate(cols):
            for j, tgt in enumerate(cols):
                if adj[i, j] > 0 and src != tgt:
                    # Skip the original X→Y (already in base dataset)
                    if src == "X" and tgt == "Y":
                        continue

                    # Proper column swap: src↔X, tgt↔Y
                    rename_map = {}
                    for c in cols:
                        if c == src:
                            rename_map[c] = "X"
                        elif c == tgt:
                            rename_map[c] = "Y"
                        elif c == "X":
                            rename_map[c] = src  # old X takes src's name
                        elif c == "Y":
                            rename_map[c] = tgt  # old Y takes tgt's name
                        else:
                            rename_map[c] = c

                    new_cols = [rename_map[c] for c in cols]

                    new_df = df.copy()
                    new_df.columns = new_cols

                    new_y = y_df.copy()
                    new_y.columns = new_cols
                    new_y.index = new_cols

                    aug_key = f"aug_{aug_idx}"
                    aug_X[aug_key] = new_df
                    aug_y[aug_key] = new_y
                    aug_idx += 1

    print(f"  {aug_idx} augmented samples ({aug_idx / len(X_data):.1f}x)")

    # Compute features for augmented data
    aug_features = compute_features_parallel(
        aug_X, aug_y, cache_path, n_workers=n_workers,
        run_causal_discovery=run_causal_discovery,
    )

    return aug_features


# ############################################################
#
#   PART 5: TREE ENSEMBLE
#
# ############################################################

def train_tree_ensemble(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    save_dir: str = "resources/",
) -> dict:
    """
    Train XGBoost + LightGBM + CatBoost ensemble.
    Returns dict with models and feature names.
    """
    # Separate features from metadata
    meta_cols = [c for c in train_df.columns if c.startswith("_")]
    feature_cols = [c for c in train_df.columns if not c.startswith("_")]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["_label"].values.astype(int)

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    if val_df is not None:
        X_val = val_df[feature_cols].values.astype(np.float32)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = val_df["_label"].values.astype(int)
    else:
        X_val, y_val = None, None

    # Class weights
    class_counts = np.bincount(y_train, minlength=N_CLASSES).astype(float)
    sample_weights = 1.0 / (class_counts[y_train] + 1.0)
    sample_weights = sample_weights / sample_weights.mean()

    models = {}
    predictions = {}

    # --- XGBoost ---
    print("\n=== Training XGBoost ===")
    try:
        import xgboost as xgb

        for seed_idx, seed in enumerate([42, 77, 2]):
            params = XGB_PARAMS.copy()
            params["random_state"] = seed

            model = xgb.XGBClassifier(**params)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=eval_set,
                verbose=100,
            )
            models[f"xgb_{seed}"] = model
            if X_val is not None:
                pred = model.predict_proba(X_val)
                predictions[f"xgb_{seed}"] = pred
                acc = accuracy_score(y_val, pred.argmax(axis=1))
                print(f"  XGBoost (seed={seed}) val acc: {acc:.4f}")
    except ImportError:
        print("  XGBoost not available, skipping")

    # --- LightGBM ---
    print("\n=== Training LightGBM ===")
    try:
        import lightgbm as lgb

        for seed_idx, seed in enumerate([42, 77, 2]):
            params = LGBM_PARAMS.copy()
            params["random_state"] = seed

            model = lgb.LGBMClassifier(**params)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            callbacks = [lgb.early_stopping(20), lgb.log_evaluation(100)] if eval_set else None
            model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=eval_set,
                callbacks=callbacks,
            )
            models[f"lgbm_{seed}"] = model
            if X_val is not None:
                pred = model.predict_proba(X_val)
                predictions[f"lgbm_{seed}"] = pred
                acc = accuracy_score(y_val, pred.argmax(axis=1))
                print(f"  LightGBM (seed={seed}) val acc: {acc:.4f}")
    except ImportError:
        print("  LightGBM not available, skipping")

    # --- CatBoost ---
    print("\n=== Training CatBoost ===")
    try:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=100,
            eval_metric="MultiClass",
            task_type="CPU",
            thread_count=48,
        )
        eval_pool = (X_val, y_val) if X_val is not None else None
        model.fit(X_train, y_train, eval_set=eval_pool, sample_weight=sample_weights)
        models["catboost"] = model
        if X_val is not None:
            pred = model.predict_proba(X_val)
            predictions["catboost"] = pred
            acc = accuracy_score(y_val, pred.argmax(axis=1))
            print(f"  CatBoost val acc: {acc:.4f}")
    except ImportError:
        print("  CatBoost not available, skipping")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "tree_models.pkl"), "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols}, f)

    if predictions and X_val is not None:
        # Average ensemble
        avg_pred = np.mean(list(predictions.values()), axis=0)
        avg_acc = accuracy_score(y_val, avg_pred.argmax(axis=1))
        print(f"\n  === Tree Ensemble val acc: {avg_acc:.4f} ===")

        # Per-class
        for c in range(N_CLASSES):
            mask = y_val == c
            if mask.sum() > 0:
                class_acc = (avg_pred[mask].argmax(axis=1) == c).mean()
                print(f"    {CLASS_NAMES[c]:20s}: {class_acc:.4f} ({mask.sum()} samples)")

    return {"models": models, "feature_cols": feature_cols, "predictions": predictions}


# ############################################################
#
#   PART 6: GNN REFINEMENT
#
#   Takes initial predictions (from trees + conv1d) and refines
#   them by reasoning about graph consistency.
#
#   Key idea: If variable A is predicted as "Mediator" (X→A→Y)
#   and variable B is predicted as "Cause of X" (B→X), then
#   B should also influence Y through A. The GNN propagates
#   such constraints.
#
# ############################################################

class GraphRefinementGNN(nn.Module):
    """
    GNN that takes initial class probabilities + features for all variables
    in a sample and refines them by message passing.

    Each node = one variable (non-X/Y). X and Y are special nodes.
    Edges = all pairs, weighted by predicted adjacency.
    """
    def __init__(self, n_feat: int, n_classes: int = N_CLASSES,
                 hidden: int = GNN_HIDDEN, n_layers: int = GNN_LAYERS,
                 n_heads: int = GNN_HEADS):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads

        # Input: initial class probs (8) + selected features
        self.input_proj = nn.Sequential(
            nn.Linear(n_feat + n_classes, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        # X and Y node embeddings
        self.x_emb = nn.Parameter(torch.randn(hidden))
        self.y_emb = nn.Parameter(torch.randn(hidden))

        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gat_layers.append(GATLayer(hidden, n_heads))

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, node_features: torch.Tensor, init_probs: torch.Tensor,
                n_vars: int) -> torch.Tensor:
        """
        node_features: (n_vars, n_feat) — features for non-X/Y variables
        init_probs: (n_vars, n_classes) — initial class probabilities
        n_vars: number of non-X/Y variables

        Returns: (n_vars, n_classes) refined logits
        """
        # Concatenate features + initial probs
        x = torch.cat([node_features, init_probs], dim=-1)
        x = self.input_proj(x)  # (n_vars, hidden)

        # Add X and Y as special nodes
        x_node = self.x_emb.unsqueeze(0)  # (1, hidden)
        y_node = self.y_emb.unsqueeze(0)
        nodes = torch.cat([x_node, y_node, x], dim=0)  # (2 + n_vars, hidden)

        # Message passing
        for gat in self.gat_layers:
            nodes = gat(nodes)

        # Output for non-X/Y nodes only
        var_nodes = nodes[2:]  # (n_vars, hidden)
        return self.output_head(var_nodes)


class GATLayer(nn.Module):
    """Multi-head graph attention layer."""
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        h = self.n_heads
        hd = self.head_dim

        q = self.q(x).view(n, h, hd)
        k = self.k(x).view(n, h, hd)
        v = self.v(x).view(n, h, hd)

        attn = torch.einsum('ihd,jhd->hij', q, k) / (hd ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('hij,jhd->ihd', attn, v).reshape(n, d)
        out = self.o(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


# ############################################################
#
#   PART 7: STACKING ENSEMBLE
#
#   Combines conv1d model, tree ensemble, and GNN predictions.
#
# ############################################################

def train_stacking_ensemble(
    val_features_df: pd.DataFrame,
    conv1d_probs: Optional[np.ndarray],
    tree_probs: dict,
    gnn_probs: Optional[np.ndarray] = None,
    save_path: str = "resources/stacker.pkl",
) -> dict:
    """
    Train a stacking ensemble that combines all model predictions.

    Inputs per sample:
      - XGBoost/LightGBM/CatBoost class probabilities (8 dims each)
      - Conv1d class probabilities (8 dims, if available)
      - GNN refined probabilities (8 dims, if available)
    """
    y_val = val_features_df["_label"].values.astype(int)

    # Build stacking features
    stack_features = []

    # Tree predictions
    for name, probs in tree_probs.items():
        stack_features.append(probs)

    # Conv1d predictions (if available)
    if conv1d_probs is not None:
        stack_features.append(conv1d_probs)

    # GNN predictions (if available)
    if gnn_probs is not None:
        stack_features.append(gnn_probs)

    # Concatenate
    X_stack = np.hstack(stack_features)
    print(f"Stacking features shape: {X_stack.shape}")

    # Train logistic regression stacker with CV
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(y_val), N_CLASSES))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_stack, y_val)):
        X_tr, X_va = X_stack[train_idx], X_stack[val_idx]
        y_tr, y_va = y_val[train_idx], y_val[val_idx]

        stacker = LogisticRegression(
            C=1.0, max_iter=1000, multi_class="multinomial",
            solver="lbfgs", random_state=42,
        )
        stacker.fit(X_tr, y_tr)
        oof_preds[val_idx] = stacker.predict_proba(X_va)

        fold_acc = accuracy_score(y_va, oof_preds[val_idx].argmax(axis=1))
        print(f"  Fold {fold}: {fold_acc:.4f}")

    # Overall OOF accuracy
    oof_acc = accuracy_score(y_val, oof_preds.argmax(axis=1))
    print(f"\n  === Stacking OOF accuracy: {oof_acc:.4f} ===")

    # Per-class
    for c in range(N_CLASSES):
        mask = y_val == c
        if mask.sum() > 0:
            class_acc = (oof_preds[mask].argmax(axis=1) == c).mean()
            print(f"    {CLASS_NAMES[c]:20s}: {class_acc:.4f}")

    # Retrain on all data
    final_stacker = LogisticRegression(
        C=1.0, max_iter=1000, multi_class="multinomial",
        solver="lbfgs", random_state=42,
    )
    final_stacker.fit(X_stack, y_val)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(final_stacker, f)

    return {"stacker": final_stacker, "oof_acc": oof_acc}


# ############################################################
#
#   PART 8: MAIN PIPELINE
#
# ############################################################

def main():
    parser = argparse.ArgumentParser(description="v13 Full-Stack Causal Discovery")
    parser.add_argument("--stage", default="all",
                        choices=["features", "trees", "gnn", "stack", "all", "eval"])
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    parser.add_argument("--n_workers", type=int, default=48)
    parser.add_argument("--no_causal_discovery", action="store_true",
                        help="Skip running PC/LiNGAM/NOTEARS/GES (faster but fewer features)")
    parser.add_argument("--xy_aug", action="store_true",
                        help="Use XY remap augmentation for tree training")
    parser.add_argument("--conv1d_probs", default=None,
                        help="Path to conv1d model's validation probabilities (npy)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    run_cd = not args.no_causal_discovery

    # Load data
    print("Loading data...")
    X_train = pd.read_pickle(os.path.join(args.data_dir, "X_train.pickle"))
    y_train = pd.read_pickle(os.path.join(args.data_dir, "y_train.pickle"))
    X_test = pd.read_pickle(os.path.join(args.data_dir, "X_test_reduced.pickle"))

    if args.stage in ("features", "all"):
        print("\n" + "=" * 60)
        print("STAGE 1: Feature Computation")
        print("=" * 60)

        tag = "cd" if run_cd else "nocd"

        # Base features
        train_features = compute_features_parallel(
            X_train, y_train,
            os.path.join(args.cache_dir, f"train_features_{tag}.pkl"),
            n_workers=args.n_workers,
            run_causal_discovery=run_cd,
        )

        test_features = compute_features_parallel(
            X_test, None,
            os.path.join(args.cache_dir, f"test_features_{tag}.pkl"),
            n_workers=args.n_workers,
            run_causal_discovery=run_cd,
        )

        # XY augmentation
        if args.xy_aug:
            aug_features = compute_augmented_features(
                X_train, y_train,
                os.path.join(args.cache_dir, f"train_features_aug_{tag}.pkl"),
                n_workers=args.n_workers,
                run_causal_discovery=run_cd,
            )
            # Combine
            train_features = pd.concat([train_features, aug_features], ignore_index=True)
            train_features.to_pickle(os.path.join(args.cache_dir, f"train_features_combined_{tag}.pkl"))

        n_features = len([c for c in train_features.columns if not c.startswith("_")])
        print(f"\nTotal: {len(train_features)} training rows, {n_features} features")

    if args.stage in ("trees", "all"):
        print("\n" + "=" * 60)
        print("STAGE 2: Tree Ensemble Training")
        print("=" * 60)

        tag = "cd" if run_cd else "nocd"
        combined_path = os.path.join(args.cache_dir, f"train_features_combined_{tag}.pkl")
        base_path = os.path.join(args.cache_dir, f"train_features_{tag}.pkl")
        feat_path = combined_path if os.path.exists(combined_path) else base_path

        train_features = pd.read_pickle(feat_path)

        # Split for validation
        # Use sample keys for stratified split (keep samples together)
        sample_keys = train_features["_sample_key"].unique().tolist()
        train_keys, val_keys = train_test_split(sample_keys, test_size=0.1, random_state=42)

        train_mask = train_features["_sample_key"].isin(train_keys)
        val_mask = train_features["_sample_key"].isin(val_keys)

        train_df = train_features[train_mask].copy()
        val_df = train_features[val_mask].copy()

        result = train_tree_ensemble(train_df, val_df, save_dir="resources/")

    if args.stage in ("gnn", "all"):
        print("\n" + "=" * 60)
        print("STAGE 3: GNN Refinement")
        print("=" * 60)

        tag = "cd" if run_cd else "nocd"
        combined_path = os.path.join(args.cache_dir, f"train_features_combined_{tag}.pkl")
        base_path = os.path.join(args.cache_dir, f"train_features_{tag}.pkl")
        feat_path = combined_path if os.path.exists(combined_path) else base_path
        train_features = pd.read_pickle(feat_path)

        # Load tree models for initial predictions
        tree_data = pickle.load(open("resources/tree_models.pkl", "rb"))
        feature_cols = tree_data["feature_cols"]

        # Split by sample (same split as trees)
        sample_keys = train_features["_sample_key"].unique().tolist()
        train_keys, val_keys = train_test_split(sample_keys, test_size=0.1, random_state=42)

        # Select top features for GNN input (use a manageable subset)
        # Pick CI + causal discovery features — the ones most useful for cross-variable reasoning
        gnn_feat_cols = [c for c in feature_cols if any(
            p in c for p in ["cmi_", "consensus_", "pc_class_", "ges_class_",
                             "mi_", "pcorr", "hsic", "dcor", "anm_asym",
                             "pearson", "n_variables", "explaining_away",
                             "lingam_", "notears_", "reg_r2", "rank_corr"]
        )]
        if not gnn_feat_cols:
            gnn_feat_cols = feature_cols[:50]  # fallback
        n_gnn_feat = len(gnn_feat_cols)
        print(f"  GNN input features: {n_gnn_feat}")

        # Build per-sample data structures
        def build_gnn_samples(df, keys, tree_models, feat_cols, gnn_fcols):
            """Group rows by sample and build (features, init_probs, labels) per sample."""
            samples = []
            X_all = df[feat_cols].values.astype(np.float32)
            X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

            # Get tree ensemble avg probs for all rows
            all_probs = []
            for name, model in tree_models.items():
                all_probs.append(model.predict_proba(X_all))
            avg_probs = np.mean(all_probs, axis=0)  # (total_rows, 8)

            gnn_feat_idx = [feat_cols.index(c) for c in gnn_fcols if c in feat_cols]

            for key in keys:
                mask = df["_sample_key"] == key
                if mask.sum() == 0:
                    continue
                row_indices = np.where(mask.values)[0]
                n_vars = len(row_indices)

                node_feat = X_all[row_indices][:, gnn_feat_idx]  # (K, n_gnn_feat)
                init_probs = avg_probs[row_indices]               # (K, 8)

                labels = None
                if "_label" in df.columns:
                    labels = df.iloc[row_indices]["_label"].values.astype(int)

                samples.append({
                    "node_feat": node_feat,
                    "init_probs": init_probs,
                    "labels": labels,
                    "n_vars": n_vars,
                    "key": key,
                })
            return samples

        train_samples = build_gnn_samples(
            train_features[train_features["_sample_key"].isin(train_keys)],
            train_keys, tree_data["models"], feature_cols, gnn_feat_cols)
        val_samples = build_gnn_samples(
            train_features[train_features["_sample_key"].isin(val_keys)],
            val_keys, tree_data["models"], feature_cols, gnn_feat_cols)

        print(f"  Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

        # Train GNN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gnn = GraphRefinementGNN(n_feat=n_gnn_feat).to(device)
        optimizer = torch.optim.AdamW(gnn.parameters(), lr=GNN_LR, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=GNN_EPOCHS)

        # Class weights
        all_labels = np.concatenate([s["labels"] for s in train_samples if s["labels"] is not None])
        counts = np.bincount(all_labels, minlength=N_CLASSES).astype(float) + 1.0
        class_w = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
        class_w = class_w / class_w.sum() * N_CLASSES
        loss_fn = nn.CrossEntropyLoss(weight=class_w)

        best_val_acc = 0.0
        for epoch in range(GNN_EPOCHS):
            # --- Train ---
            gnn.train()
            np.random.shuffle(train_samples)
            train_loss, train_correct, train_total = 0.0, 0, 0

            for s in train_samples:
                if s["labels"] is None:
                    continue
                nf = torch.tensor(s["node_feat"], dtype=torch.float32).to(device)
                ip = torch.tensor(s["init_probs"], dtype=torch.float32).to(device)
                lab = torch.tensor(s["labels"], dtype=torch.long).to(device)

                logits = gnn(nf, ip, s["n_vars"])
                loss = loss_fn(logits, lab)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(lab)
                train_correct += (logits.argmax(-1) == lab).sum().item()
                train_total += len(lab)

            scheduler.step()

            # --- Val ---
            gnn.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for s in val_samples:
                    if s["labels"] is None:
                        continue
                    nf = torch.tensor(s["node_feat"], dtype=torch.float32).to(device)
                    ip = torch.tensor(s["init_probs"], dtype=torch.float32).to(device)
                    lab = torch.tensor(s["labels"], dtype=torch.long).to(device)

                    logits = gnn(nf, ip, s["n_vars"])
                    val_correct += (logits.argmax(-1) == lab).sum().item()
                    val_total += len(lab)

            train_acc = train_correct / max(train_total, 1)
            val_acc = val_correct / max(val_total, 1)

            if epoch % 5 == 0 or val_acc > best_val_acc:
                print(f"  Epoch {epoch:3d}  train_loss={train_loss/max(train_total,1):.4f}  "
                      f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(gnn.state_dict(), "resources/gnn_refinement.pt")

        print(f"\n  === Best GNN val acc: {best_val_acc:.4f} ===")

        # Save GNN config for inference
        with open("resources/gnn_config.json", "w") as f:
            json.dump({"n_gnn_feat": n_gnn_feat, "gnn_feat_cols": gnn_feat_cols}, f)

    if args.stage in ("stack", "all"):
        print("\n" + "=" * 60)
        print("STAGE 4: Stacking Ensemble")
        print("=" * 60)

        # Load tree predictions
        tree_data = pickle.load(open("resources/tree_models.pkl", "rb"))

        tag = "cd" if run_cd else "nocd"
        base_path = os.path.join(args.cache_dir, f"train_features_{tag}.pkl")
        train_features = pd.read_pickle(base_path)

        # Get val split
        sample_keys = train_features["_sample_key"].unique().tolist()
        _, val_keys = train_test_split(sample_keys, test_size=0.1, random_state=42)
        val_df = train_features[train_features["_sample_key"].isin(val_keys)]

        feature_cols = tree_data["feature_cols"]
        X_val = val_df[feature_cols].values.astype(np.float32)
        X_val = np.nan_to_num(X_val)

        tree_probs = {}
        for name, model in tree_data["models"].items():
            tree_probs[name] = model.predict_proba(X_val)

        # Load conv1d predictions if available
        conv1d_probs = None
        if args.conv1d_probs and os.path.exists(args.conv1d_probs):
            conv1d_probs = np.load(args.conv1d_probs)
            print(f"Loaded conv1d predictions: {conv1d_probs.shape}")

        # Load GNN predictions if available
        gnn_probs = None
        gnn_config_path = "resources/gnn_config.json"
        gnn_model_path = "resources/gnn_refinement.pt"
        if os.path.exists(gnn_config_path) and os.path.exists(gnn_model_path):
            print("Generating GNN predictions for stacking...")
            with open(gnn_config_path, "r") as f:
                gnn_cfg = json.load(f)
            gnn_feat_cols = gnn_cfg["gnn_feat_cols"]
            n_gnn_feat = gnn_cfg["n_gnn_feat"]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            gnn = GraphRefinementGNN(n_feat=n_gnn_feat).to(device)
            gnn.load_state_dict(torch.load(gnn_model_path, map_location=device))
            gnn.eval()

            # Get tree avg probs for GNN input
            tree_avg = np.mean(list(tree_probs.values()), axis=0)
            gnn_feat_idx = [feature_cols.index(c) for c in gnn_feat_cols if c in feature_cols]

            # Run GNN per sample
            gnn_probs = np.zeros((len(val_df), N_CLASSES), dtype=np.float32)
            with torch.no_grad():
                for key in val_df["_sample_key"].unique():
                    mask = val_df["_sample_key"] == key
                    row_idx = np.where(mask.values)[0]

                    nf = torch.tensor(X_val[row_idx][:, gnn_feat_idx], dtype=torch.float32).to(device)
                    ip = torch.tensor(tree_avg[row_idx], dtype=torch.float32).to(device)
                    logits = gnn(nf, ip, len(row_idx))
                    gnn_probs[row_idx] = F.softmax(logits, dim=-1).cpu().numpy()

            print(f"  GNN predictions shape: {gnn_probs.shape}")

        train_stacking_ensemble(val_df, conv1d_probs, tree_probs, gnn_probs=gnn_probs)

    if args.stage == "eval":
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)

        # Load everything and evaluate
        tag = "cd" if run_cd else "nocd"
        test_features = pd.read_pickle(os.path.join(args.cache_dir, f"test_features_{tag}.pkl"))
        tree_data = pickle.load(open("resources/tree_models.pkl", "rb"))

        feature_cols = tree_data["feature_cols"]
        X_test_feat = test_features[feature_cols].values.astype(np.float32)
        X_test_feat = np.nan_to_num(X_test_feat)

        # Get tree predictions
        all_preds = []
        for name, model in tree_data["models"].items():
            pred = model.predict_proba(X_test_feat)
            all_preds.append(pred)

        # Average
        avg_pred = np.mean(all_preds, axis=0)
        test_preds = avg_pred.argmax(axis=1)

        # If ground truth available
        y_test_path = os.path.join(args.data_dir, "y_test_reduced.pickle")
        if os.path.exists(y_test_path):
            y_test_data = pd.read_pickle(y_test_path)

            correct = 0
            total = 0
            per_class_correct = np.zeros(N_CLASSES)
            per_class_total = np.zeros(N_CLASSES)

            for idx, row in test_features.iterrows():
                pred = test_preds[idx]
                true = None

                sample_key = row.get("_sample_key")
                var_name = row.get("_variable")

                if sample_key in y_test_data:
                    y_df = y_test_data[sample_key]
                    adj_df = pd.DataFrame(y_df.values, index=list(y_df.columns), columns=list(y_df.columns))
                    labels = get_labels(adj_df)
                    true = labels.get(var_name)

                if true is not None:
                    total += 1
                    per_class_total[true] += 1
                    if pred == true:
                        correct += 1
                        per_class_correct[true] += 1

            if total > 0:
                print(f"\nOverall accuracy: {correct/total:.4f} ({correct}/{total})")
                for c in range(N_CLASSES):
                    if per_class_total[c] > 0:
                        acc = per_class_correct[c] / per_class_total[c]
                        print(f"  {CLASS_NAMES[c]:20s}: {acc:.4f} ({int(per_class_correct[c])}/{int(per_class_total[c])})")


if __name__ == "__main__":
    main()
