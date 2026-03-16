"""
Parallelized local evaluation — node-level balanced accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import networkx as nx


# --- Graph utilities (must be top-level for pickling) ---
def graph_nodes_representation(graph, nodelist):
    return tuple(nx.adjacency_matrix(graph, nodelist=nodelist).todense().flatten())


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
        graph_nodes_representation(g, nodelist): l
        for g, l in graph_label.items()
    }
    return graph_label, adjacency_label


def get_labels(adj_df, adjacency_label):
    result = {}
    for v in adj_df.columns.drop(["X", "Y"]):
        sub = adj_df.loc[[v, "X", "Y"], [v, "X", "Y"]]
        key = tuple(sub.values.flatten())
        result[v] = adjacency_label[key]
    return result


def _eval_single(args):
    """Process one graph: reconstruct adj from flat predictions, extract labels."""
    name, nodes, pred_lookup, true_lookup, adjacency_label = args
    p = len(nodes)

    pred_labels, true_labels = [], []
    for source, lookup in [("pred", pred_lookup), ("true", true_lookup)]:
        adj = np.zeros((p, p), dtype=int)
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                adj[i, j] = lookup.get(f"{name}_{ni}_{nj}", 0)
        A = pd.DataFrame(adj, columns=nodes, index=nodes)
        labels = get_labels(A, adjacency_label)
        if source == "pred":
            pred_labels.extend(labels.values())
        else:
            true_labels.extend(labels.values())

    return pred_labels, true_labels


def evaluate_predictions(X_test, y_pred_path, y_true_path, n_workers=None):
    """
    Parallel evaluation of predictions vs ground truth.

    Args:
        X_test: dict of {name: DataFrame} — just need column names
        y_pred_path: path to prediction parquet
        y_true_path: path to ground truth parquet
        n_workers: number of processes (default: cpu_count - 1)
    """
    import multiprocessing as mp
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    # Load and index predictions
    y_pred_df = pd.read_parquet(y_pred_path)
    y_true_df = pd.read_parquet(y_true_path)

    # Build fast lookup dicts (id -> prediction value)
    pred_lookup = dict(zip(y_pred_df.iloc[:, 0], y_pred_df["prediction"]))
    true_lookup = dict(zip(y_true_df.iloc[:, 0], y_true_df["prediction"]))

    _, adjacency_label = create_graph_label()

    # Build args
    args = [
        (name, list(X_test[name].columns), pred_lookup, true_lookup, adjacency_label)
        for name in X_test.keys()
    ]

    print(f"Evaluating {len(args)} graphs with {n_workers} workers...")
    all_pred, all_true = [], []

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for pred_labels, true_labels in tqdm(
                pool.map(_eval_single, args, chunksize=64), total=len(args)
            ):
                all_pred.extend(pred_labels)
                all_true.extend(true_labels)
    else:
        for a in tqdm(args):
            pred_labels, true_labels = _eval_single(a)
            all_pred.extend(pred_labels)
            all_true.extend(true_labels)

    score = balanced_accuracy_score(all_true, all_pred)

    # Per-class breakdown
    y_pred_s = pd.Series(all_pred)
    y_true_s = pd.Series(all_true)
    print("\nPer-class accuracy:")
    for label in sorted(y_true_s.unique()):
        mask = y_true_s == label
        acc = np.mean(y_pred_s[mask] == label)
        print(f"  {label:25s}: {acc:.4f}  (n={mask.sum()})")

    print(f"\nBalanced Accuracy: {score:.4f}")
    return score


if __name__ == "__main__":
    # --- Usage ---
    import crunch
    crunch = crunch.load_notebook()
    X_train, y_train, X_test = crunch.load_data()

    score = evaluate_predictions(
        X_test,
        y_pred_path="prediction/prediction_20_b16_lr1e3.parquet",
        y_true_path="data/example_prediction_reduced.parquet",
    )