"""
v10_lgbm.py — ADIA Causal Discovery
  Gradient Boosted Trees with rich scalar features per variable.

Different paradigm from top-1 DL approach. Inspired by 3rd/4th/5th place
tree-based solutions, but using OUR preprocessing (multi-bandwidth kernel
regression + ANM residuals) as feature source.

Features per variable v (relative to X, Y):
  Group 1: Correlation (Pearson, Spearman, partial corr, distance corr)
  Group 2: Kernel regression coefficient stats (mean/std/skew/kurtosis per bw per edge)
  Group 3: ANM residual stats (mean/std/skew/kurtosis per bw per edge)
  Group 4: Mutual information / CMI
  Group 5: Graph-level context (n_vars, corr(X,Y), etc.)

Training: XY remap augmentation (~11x), LGBM with class_weight="balanced".
Go/no-go on base 25K first, then XY aug.

Usage:
    python v10_lgbm.py
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import typing
import gc
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

import networkx as nx
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score

# ============================================================
# Configuration
# ============================================================
N_KERNEL: int = 1000
N_CLASSES: int = 8
CLASS_NAMES: list[str] = [
    "Confounder", "Collider", "Mediator",
    "Cause of X", "Cause of Y",
    "Consequence of X", "Consequence of Y",
    "Independent",
]
BANDWIDTHS: list[float] = [0.2, 0.5, 1.0]
LOCAL_CACHE_DIR: str = "dataset_cache/"


# ============================================================
# Graph Utilities
# ============================================================
def graph_nodes_representation(graph: nx.DiGraph, nodelist: list[str]) -> tuple:
    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=nodelist).todense()
    return tuple(adjacency_matrix.flatten())


def create_graph_label() -> tuple[dict, dict]:
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


_ADJACENCY_LABEL = None
def get_adjacency_label() -> dict:
    global _ADJACENCY_LABEL
    if _ADJACENCY_LABEL is None:
        _, _ADJACENCY_LABEL = create_graph_label()
    return _ADJACENCY_LABEL


def get_labels(adjacency_matrix: pd.DataFrame, adjacency_label: dict) -> dict:
    result = {}
    for variable in adjacency_matrix.columns.drop(["X", "Y"]):
        submatrix = adjacency_matrix.loc[
            [variable, "X", "Y"], [variable, "X", "Y"]
        ]
        key = tuple(submatrix.values.flatten())
        result[variable] = adjacency_label[key]
    return result


# ============================================================
# Kernel Regression (reuse from v8b)
# ============================================================
def compute_multivariate_kernel_coefficients(
    data: np.ndarray, sub_idx: np.ndarray, bandwidth: float = 0.5,
) -> tuple[dict, dict]:
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

    coeff_map = {}
    resid_map = {}
    for j in range(p):
        c_all, other_cols = all_coeffs[j]
        for idx_in_other, k in enumerate(other_cols):
            coeff_sub = c_all[:, idx_in_other + 1]
            coeff_map[(k, j)] = coeff_sub[nearest].astype(np.float32)
        c_nn = c_all[nearest]
        X_full = np.concatenate([np.ones((N, 1)), data[:, other_cols]], axis=1)
        y_hat = np.sum(c_nn * X_full, axis=1)
        resid_map[j] = (data[:, j] - y_hat).astype(np.float32)

    return coeff_map, resid_map


# ============================================================
# Feature Extraction
# ============================================================
def _safe_stats(arr: np.ndarray) -> dict:
    """Compute mean/std/skew/kurtosis safely."""
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0}
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "skew": float(sp_stats.skew(arr)),
        "kurt": float(sp_stats.kurtosis(arr)),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    r, _ = sp_stats.spearmanr(a, b)
    return float(r) if np.isfinite(r) else 0.0


def _partial_corr(data: np.ndarray, i: int, j: int) -> float:
    """Partial correlation between i and j given all other variables."""
    p = data.shape[1]
    cov = np.cov(data.T)
    cov += 1e-6 * np.eye(p)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.0
    d = np.sqrt(max(prec[i, i] * prec[j, j], 1e-20))
    return float(-prec[i, j] / d)


def _distance_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Simple distance correlation implementation."""
    n = len(a)
    if n < 4:
        return 0.0
    a_dist = np.abs(a[:, None] - a[None, :])
    b_dist = np.abs(b[:, None] - b[None, :])
    a_mean_row = a_dist.mean(axis=1, keepdims=True)
    a_mean_col = a_dist.mean(axis=0, keepdims=True)
    a_mean = a_dist.mean()
    A = a_dist - a_mean_row - a_mean_col + a_mean
    b_mean_row = b_dist.mean(axis=1, keepdims=True)
    b_mean_col = b_dist.mean(axis=0, keepdims=True)
    b_mean = b_dist.mean()
    B = b_dist - b_mean_row - b_mean_col + b_mean
    dcov2 = (A * B).mean()
    dvar_a = (A * A).mean()
    dvar_b = (B * B).mean()
    if dvar_a < 1e-20 or dvar_b < 1e-20:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvar_a * dvar_b)))


def extract_features_for_variable(
    data: np.ndarray, cols: list[str], v_name: str,
    coeff_maps: list[dict], resid_maps: list[dict],
) -> dict:
    """Extract all scalar features for one variable v relative to X, Y."""
    col2idx = {c: i for i, c in enumerate(cols)}
    vi = col2idx[v_name]
    xi = col2idx["X"]
    yi = col2idx["Y"]
    p = len(cols)
    N = data.shape[0]

    feats: dict[str, float] = {}

    # ---- Group 1: Correlation features ----
    feats["pearson_vX"] = _safe_corr(data[:, vi], data[:, xi])
    feats["pearson_vY"] = _safe_corr(data[:, vi], data[:, yi])
    feats["pearson_XY"] = _safe_corr(data[:, xi], data[:, yi])
    feats["abs_pearson_vX"] = abs(feats["pearson_vX"])
    feats["abs_pearson_vY"] = abs(feats["pearson_vY"])

    feats["spearman_vX"] = _safe_spearman(data[:, vi], data[:, xi])
    feats["spearman_vY"] = _safe_spearman(data[:, vi], data[:, yi])

    # Partial correlations
    feats["pcorr_vX_given_rest"] = _partial_corr(data, vi, xi)
    feats["pcorr_vY_given_rest"] = _partial_corr(data, vi, yi)
    feats["pcorr_XY_given_rest"] = _partial_corr(data, xi, yi)

    # Distance correlation (subsample for speed)
    sub = np.random.choice(N, min(500, N), replace=False)
    feats["dcor_vX"] = _distance_correlation(data[sub, vi], data[sub, xi])
    feats["dcor_vY"] = _distance_correlation(data[sub, vi], data[sub, yi])

    # Correlation with other variables (stats)
    other_corrs = []
    for k in range(p):
        if k in (vi, xi, yi):
            continue
        other_corrs.append(abs(_safe_corr(data[:, vi], data[:, k])))
    if other_corrs:
        feats["other_corr_max"] = max(other_corrs)
        feats["other_corr_mean"] = np.mean(other_corrs)
        feats["other_corr_std"] = np.std(other_corrs)
    else:
        feats["other_corr_max"] = 0.0
        feats["other_corr_mean"] = 0.0
        feats["other_corr_std"] = 0.0

    # ---- Group 2: Kernel regression coefficient stats ----
    # For edges v→X, v→Y, X→v, Y→v at each bandwidth
    edge_pairs = [
        ("vX", vi, xi), ("vY", vi, yi),
        ("Xv", xi, vi), ("Yv", yi, vi),
    ]
    for bw_idx, bw in enumerate(BANDWIDTHS):
        cm = coeff_maps[bw_idx]
        rm = resid_maps[bw_idx]
        for edge_name, src, dst in edge_pairs:
            key = (src, dst)
            if key in cm:
                s = _safe_stats(cm[key])
                for stat_name, val in s.items():
                    feats[f"kern_{edge_name}_bw{bw}_{stat_name}"] = val
                # Also: abs mean, range
                arr = cm[key]
                feats[f"kern_{edge_name}_bw{bw}_absmean"] = float(np.mean(np.abs(arr)))
                feats[f"kern_{edge_name}_bw{bw}_range"] = float(np.max(arr) - np.min(arr))

    # ---- Group 3: ANM residual stats ----
    for bw_idx, bw in enumerate(BANDWIDTHS):
        rm = resid_maps[bw_idx]
        for edge_name, src, dst in edge_pairs:
            if dst in rm:
                resid = rm[dst]
                s = _safe_stats(resid)
                for stat_name, val in s.items():
                    feats[f"anm_{edge_name}_bw{bw}_{stat_name}"] = val
                # Correlation of residual with source (ANM asymmetry test)
                feats[f"anm_{edge_name}_bw{bw}_corr_src"] = _safe_corr(resid, data[:, src])
                feats[f"anm_{edge_name}_bw{bw}_abs_corr_src"] = abs(feats[f"anm_{edge_name}_bw{bw}_corr_src"])

    # ---- Group 4: Regression asymmetry (simple linear) ----
    for edge_name, src, dst in edge_pairs:
        xs = data[:, src]
        xd = data[:, dst]
        var_s = np.var(xs)
        if var_s > 1e-10:
            b = np.cov(xs, xd)[0, 1] / var_s
            resid = xd - (xd.mean() + b * (xs - xs.mean()))
            feats[f"linreg_{edge_name}_resid_corr"] = _safe_corr(resid, xs)
            feats[f"linreg_{edge_name}_r2"] = float(1.0 - np.var(resid) / max(np.var(xd), 1e-10))
        else:
            feats[f"linreg_{edge_name}_resid_corr"] = 0.0
            feats[f"linreg_{edge_name}_r2"] = 0.0

    # ---- Group 5: Graph-level context ----
    feats["n_vars"] = float(p)
    feats["n_other"] = float(p - 2)  # excluding X, Y

    # ---- Group 6: Asymmetry features (key for direction detection) ----
    feats["pearson_diff_XY"] = feats["abs_pearson_vX"] - feats["abs_pearson_vY"]
    feats["dcor_diff_XY"] = feats["dcor_vX"] - feats["dcor_vY"]
    feats["pcorr_diff_XY"] = abs(feats["pcorr_vX_given_rest"]) - abs(feats["pcorr_vY_given_rest"])

    # Kernel asymmetry: compare v→X vs X→v
    for bw in BANDWIDTHS:
        km_vX = feats.get(f"kern_vX_bw{bw}_absmean", 0)
        km_Xv = feats.get(f"kern_Xv_bw{bw}_absmean", 0)
        km_vY = feats.get(f"kern_vY_bw{bw}_absmean", 0)
        km_Yv = feats.get(f"kern_Yv_bw{bw}_absmean", 0)
        feats[f"kern_asym_X_bw{bw}"] = km_vX - km_Xv
        feats[f"kern_asym_Y_bw{bw}"] = km_vY - km_Yv

    # ANM asymmetry
    for bw in BANDWIDTHS:
        ac_vX = feats.get(f"anm_vX_bw{bw}_abs_corr_src", 0)
        ac_Xv = feats.get(f"anm_Xv_bw{bw}_abs_corr_src", 0)
        ac_vY = feats.get(f"anm_vY_bw{bw}_abs_corr_src", 0)
        ac_Yv = feats.get(f"anm_Yv_bw{bw}_abs_corr_src", 0)
        feats[f"anm_asym_X_bw{bw}"] = ac_vX - ac_Xv
        feats[f"anm_asym_Y_bw{bw}"] = ac_vY - ac_Yv

    return feats


# ============================================================
# Per-graph feature builder
# ============================================================
def _build_features_one_graph(args: tuple) -> list[dict]:
    """Build features for all variables in one graph. Returns list of rows."""
    df, y_df = args
    cols = list(df.columns)
    data = df.values.astype(np.float32)
    N, p = data.shape

    # Compute kernel regression + residuals at all bandwidths
    n_sub = min(N_KERNEL, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)
    coeff_maps = []
    resid_maps = []
    for bw in BANDWIDTHS:
        cm, rm = compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=bw)
        coeff_maps.append(cm)
        resid_maps.append(rm)

    other_nodes = [c for c in cols if c not in ("X", "Y")]
    rows = []

    # Get labels if available
    adjacency_label = get_adjacency_label()
    labels_dict = None
    if y_df is not None:
        try:
            labels_dict = get_labels(y_df, adjacency_label)
        except KeyError:
            labels_dict = None

    for v_name in other_nodes:
        feats = extract_features_for_variable(data, cols, v_name, coeff_maps, resid_maps)
        if labels_dict is not None and v_name in labels_dict:
            feats["label"] = CLASS_NAMES.index(labels_dict[v_name])
        rows.append(feats)

    return rows


def _remap_xy_names(cols, new_x, new_y):
    result = list(cols)
    pos = {c: i for i, c in enumerate(cols)}
    i_nx, i_x = pos[new_x], pos["X"]
    result[i_nx], result[i_x] = result[i_x], result[i_nx]
    pos2 = {c: i for i, c in enumerate(result)}
    i_ny, i_y = pos2[new_y], pos2["Y"]
    result[i_ny], result[i_y] = result[i_y], result[i_ny]
    return {cols[i]: result[i] for i in range(len(cols))}


def _build_features_augmented(args: tuple) -> list[dict]:
    """Build features for one graph + all XY remap augmented versions.
    Kernel regression computed ONCE per base graph."""
    df, y_df = args
    cols = list(df.columns)
    data = df.values.astype(np.float32)
    N, p = data.shape

    # Compute expensive kernel regression ONCE
    n_sub = min(N_KERNEL, N)
    sub_idx = np.random.choice(N, n_sub, replace=False)
    coeff_maps = []
    resid_maps = []
    for bw in BANDWIDTHS:
        cm, rm = compute_multivariate_kernel_coefficients(data, sub_idx, bandwidth=bw)
        coeff_maps.append(cm)
        resid_maps.append(rm)

    adjacency_label = get_adjacency_label()
    all_rows = []

    def _extract_for_graph(cur_cols, cur_y_df):
        other_nodes = [c for c in cur_cols if c not in ("X", "Y")]
        labels_dict = None
        if cur_y_df is not None:
            try:
                labels_dict = get_labels(cur_y_df, adjacency_label)
            except KeyError:
                return []
        rows = []
        for v_name in other_nodes:
            feats = extract_features_for_variable(data, cur_cols, v_name, coeff_maps, resid_maps)
            if labels_dict is not None and v_name in labels_dict:
                feats["label"] = CLASS_NAMES.index(labels_dict[v_name])
            rows.append(feats)
        return rows

    # Base
    all_rows.extend(_extract_for_graph(cols, y_df))

    # XY remap augmentation
    if y_df is not None:
        adj_np = y_df.values
        col_idx = {c: i for i, c in enumerate(y_df.columns)}
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
                all_rows.extend(_extract_for_graph(new_cols, new_y_df))

    return all_rows


# ============================================================
# Build dataset
# ============================================================
def build_feature_dataset(
    X_dict: dict, y_dict: dict | None,
    n_workers: int = 40, augment: bool = False,
    cache_path: str | None = None,
) -> pd.DataFrame:
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_pickle(cache_path)

    keys = list(X_dict.keys())
    if augment and y_dict is not None:
        args = [(X_dict[k], y_dict[k]) for k in keys]
        worker_fn = _build_features_augmented
        print(f"Building augmented features ({len(args)} base graphs, {n_workers} workers)...")
    else:
        args = [(X_dict[k], y_dict[k] if y_dict else None) for k in keys]
        worker_fn = _build_features_one_graph
        print(f"Building features ({len(args)} graphs, {n_workers} workers)...")

    all_rows = []
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=n_workers) as pool:
        for result_list in tqdm(
            pool.imap_unordered(worker_fn, args, chunksize=4),
            total=len(args),
        ):
            all_rows.extend(result_list)

    print(f"Total rows: {len(all_rows)}")
    df = pd.DataFrame(all_rows)
    df = df.fillna(0.0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_pickle(cache_path)
        print(f"Cached to {cache_path}")

    return df


# ============================================================
# Train & Evaluate
# ============================================================
def train_lgbm(
    train_df: pd.DataFrame, n_estimators: int = 5000,
) -> lgb.LGBMClassifier:
    feature_cols = [c for c in train_df.columns if c != "label"]
    X = train_df[feature_cols].values
    y = train_df["label"].values.astype(int)

    print(f"Training LGBM: {X.shape[0]} samples, {X.shape[1]} features, {n_estimators} estimators")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.01,
        max_depth=8,
        num_leaves=64,
        colsample_bytree=0.5,
        subsample=0.8,
        reg_alpha=0.3,
        reg_lambda=0.3,
        class_weight="balanced",
        objective="multiclass",
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def evaluate(
    model: lgb.LGBMClassifier,
    X_test_dict: dict, y_test_dict: dict,
    n_workers: int = 40,
) -> float:
    # Build test features (no augmentation)
    test_df = build_feature_dataset(
        X_test_dict, y_test_dict, n_workers=n_workers, augment=False,
        cache_path=os.path.join(LOCAL_CACHE_DIR, "test_features_v10.pkl"),
    )

    feature_cols = [c for c in test_df.columns if c != "label"]
    X = test_df[feature_cols].values
    y_true = test_df["label"].values.astype(int)

    y_pred = model.predict(X)

    # Per-class accuracy
    print("\nPer-class accuracy:")
    accs = []
    for ci, cls in enumerate(CLASS_NAMES):
        mask = y_true == ci
        n = mask.sum()
        if n > 0:
            acc = (y_pred[mask] == ci).sum() / n
        else:
            acc = 0.0
        accs.append(acc)
        print(f"  {cls:25s}: {acc:.4f}  (n={n})")

    balanced_acc = np.mean(accs)
    print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
    return balanced_acc


# ============================================================
# Inference (for CrunchDAO)
# ============================================================
def infer_lgbm(
    model: lgb.LGBMClassifier,
    X_test_dict: dict, feature_cols: list[str],
    n_workers: int = 40,
) -> dict[str, dict[str, str]]:
    """Returns dict: graph_name -> {var_name: class_name}"""
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

    keys = list(X_test_dict.keys())
    results = {}

    for k in tqdm(keys, desc="Inference"):
        df = X_test_dict[k]
        rows = _build_features_one_graph((df, None))
        if not rows:
            continue
        feat_df = pd.DataFrame(rows).fillna(0.0)
        # Align columns
        for c in feature_cols:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        X = feat_df[feature_cols].values
        preds = model.predict(X)

        cols = list(df.columns)
        other_nodes = [c for c in cols if c not in ("X", "Y")]
        p = len(cols)
        adj = np.zeros((p, p), dtype=int)
        xi_idx = cols.index("X")
        yi_idx = cols.index("Y")
        adj[xi_idx, yi_idx] = 1

        for i, v_name in enumerate(other_nodes):
            pred_class = CLASS_NAMES[int(preds[i])]
            for (s, d) in patterns[pred_class](v_name):
                si = cols.index(s)
                di = cols.index(d)
                adj[si, di] = 1

        A = pd.DataFrame(adj, columns=cols, index=cols)
        results[k] = A

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("v10 LGBM: Scalar features + Gradient Boosted Trees")
    print(f"Bandwidths: {BANDWIDTHS}")
    print("=" * 60)

    X_train = pd.read_pickle("data/X_train.pickle")
    y_train = pd.read_pickle("data/y_train.pickle")
    print(f"Loaded {len(X_train)} training samples.")

    # --- Build features (with XY aug) ---
    train_cache = os.path.join(LOCAL_CACHE_DIR, "train_features_v10_base.pkl")
    train_df = build_feature_dataset(
        X_train, y_train, n_workers=40, augment=False,
        cache_path=train_cache,
    )
    print(f"Train features: {train_df.shape}")

    # --- Train LGBM ---
    model = train_lgbm(train_df, n_estimators=5000)

    # --- Evaluate ---
    X_test = pd.read_pickle("data/X_test_reduced.pickle")
    y_test = pd.read_pickle("data/y_test_reduced.pickle")

    feature_cols = [c for c in train_df.columns if c != "label"]
    evaluate(model, X_test, y_test, n_workers=40)

    # --- Save model ---
    os.makedirs("resources", exist_ok=True)
    with open("resources/lgbm_v10.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print("Model saved to resources/lgbm_v10.pkl")