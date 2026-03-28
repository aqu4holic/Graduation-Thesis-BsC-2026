# %%
"""
Parallelized local evaluation — node-level balanced accuracy.
"""
from matplotlib.pylab import require
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import networkx as nx
import argparse

# %%
# --- Graph utilities (must be top-level for pickling) ---
def graph_nodes_representation(graph, nodelist):
    arr = np.asarray(nx.adjacency_matrix(graph, nodelist=nodelist).todense())
    return tuple(int(x) for x in arr.flatten())


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
        key = tuple(int(x) for x in sub.values.flatten())
        result[v] = adjacency_label.get(key, "Independent")
    return result


def _eval_single(args):
    name, nodes, pred_lookup, true_adj_np, adjacency_label = args
    p = len(nodes)

    # Ground truth: directly from y_test adjacency matrix
    A_true = pd.DataFrame(true_adj_np, columns=nodes, index=nodes)
    true_labels = get_labels(A_true, adjacency_label)

    # Prediction: reconstruct from flat parquet
    adj = np.zeros((p, p), dtype=int)
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            adj[i, j] = int(pred_lookup.get(f"{name}_{ni}_{nj}", 0))
    A_pred = pd.DataFrame(adj, columns=nodes, index=nodes)
    pred_labels = get_labels(A_pred, adjacency_label)

    return list(pred_labels.values()), list(true_labels.values())


def evaluate_predictions(X_test, y_test, y_pred_path, n_workers=None):
    """
    Args:
        X_test: dict {name: DataFrame} — for column names
        y_test: dict {name: DataFrame} — ground truth adjacency matrices
        y_pred_path: path to prediction parquet
    """
    import multiprocessing as mp
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    y_pred_df = pd.read_parquet(y_pred_path)
    pred_lookup = dict(zip(y_pred_df.iloc[:, 0], y_pred_df["prediction"]))

    _, adjacency_label = create_graph_label()

    args = [
        (name, list(X_test[name].columns), pred_lookup,
         y_test[name].values.astype(int), adjacency_label)
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

    y_pred_s, y_true_s = pd.Series(all_pred), pd.Series(all_true)
    print("\nPer-class accuracy:")
    for label in sorted(y_true_s.unique()):
        mask = y_true_s == label
        acc = np.mean(y_pred_s[mask] == label)
        print(f"  {label:25s}: {acc:.4f}  (n={mask.sum()})")

    print(f"\nBalanced Accuracy: {score:.4f}")
    return score


# %%
# --- Usage ---
X_test = pd.read_pickle("data/X_test_reduced.pickle")
y_test = pd.read_pickle("data/y_test_reduced.pickle")

# # %%
# X_train = pd.read_pickle("data/X_train.pickle")
# len(X_train)

# # %%
# len(X_train), len(X_test)

# read file name from arguments
parser = argparse.ArgumentParser(description="A simple file reader")
parser.add_argument(
    "--pred_path",
    type=str,
    required=True,
    # default="prediction/v12_predictions.parquet",
    help="Path to the prediction parquet file",
)
args = parser.parse_args()

# %%
score = evaluate_predictions(
    X_test,
    y_test,
    y_pred_path=args.pred_path,
)