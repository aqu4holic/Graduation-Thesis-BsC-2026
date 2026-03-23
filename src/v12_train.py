"""
v12_train.py — Training + inference for the V12 Graph Transformer.

Key differences from v8b:
  - Dataset yields FULL GRAPHS (all nodes at once), not individual edge samples
  - Collate pads to max_n in each batch
  - Loss is cross-entropy over non-X,Y nodes only
  - XY remap aug is baked into the dataset (each GraphData has multiple XY assignments)
  - DDP: 2 GPUs, strategy="ddp", use_distributed_sampler=True (standard DataLoader)

Run:
    OPENBLAS_NUM_THREADS=1 python v12_train.py
"""

from __future__ import annotations

# @crunch/keep:on
import os
import typing
import pickle
import gc
import glob
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from tqdm.auto import tqdm

import crunch

# ─── Import local modules ─────────────────────────────────────────────────────
from v12_features import (
    GraphData,
    build_from_pickles,
    CLASS_NAMES,
    N_SUB,
    N_CHANNELS,
)
from v12_model import (
    V12Model,
    build_v12_model,
    N_NODE_FEATS,
    N_EXTRA_EDGE,
    N_CLASSES,
)

# ─── Config ───────────────────────────────────────────────────────────────────
IS_CLOUD_SUBMIT = False    # set True when submitting to CrunchDAO

N_EPOCHS      = 30
BATCH_SIZE    = 32         # graphs per GPU (full graph, not edges)
LR            = 5e-4
D_EDGE        = 64
D_MODEL       = 256
N_HEADS       = 8
N_LAYERS      = 6
DROPOUT       = 0.1
MAX_N         = 25         # max variables per graph (pad to this)

CACHE_DIR     = "dataset_cache/v12"
MODEL_TAG     = "v12_graphtransformer"

CLASS_WEIGHTS = torch.tensor([
    # Confounder, Collider, Mediator, CauseX, CauseY, ConseqX, ConseqY, Indep
    # Rough inverse frequencies from v8b experience
    3.0, 3.5, 3.5, 2.0, 2.0, 2.0, 2.0, 1.0
], dtype=torch.float32)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class GraphDataset(Dataset):
    """
    Each item is one (graph, xy_assignment) pair.
    Pre-flattens all (GraphData, x_name, y_name) into a list during __init__.
    """

    def __init__(self, shard_paths: list[str], shuffle_items: bool = True):
        self.items: list[tuple[GraphData, str, str]] = []
        for path in tqdm(shard_paths, desc="Loading shards"):
            with open(path, "rb") as f:
                graphs: list[GraphData] = pickle.load(f)
            for gd in graphs:
                for (x_name, y_name) in gd.labels_by_xy:
                    self.items.append((gd, x_name, y_name))
        if shuffle_items:
            random.shuffle(self.items)
        print(f"GraphDataset: {len(self.items)} (graph, xy) pairs loaded.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        gd, x_name, y_name = self.items[idx]
        return _build_item(gd, x_name, y_name, with_labels=True)


def _build_item(gd: GraphData, x_name: str, y_name: str, with_labels: bool) -> dict:
    """
    Convert a GraphData + XY assignment to a model-ready dict.

    Returns:
      edge_seqs   [n, n, 8, N_SUB]   float32
      extra_edges [n, n, 3]           float32
      node_feats  [n, 7]              float32
      node_types  [n]                 int64   (0=X, 1=Y, 2=other)
      mask        [n]                 bool    (all True — padding done in collate)
      node_labels [n]                 int64   (-1 for X, Y)  (only if with_labels)
      cols        list[str]
    """
    n = gd.n
    x_idx = gd.cols.index(x_name)
    y_idx = gd.cols.index(y_name)

    node_types = np.full(n, 2, dtype=np.int64)
    node_types[x_idx] = 0
    node_types[y_idx] = 1

    node_feats  = gd.get_node_features(x_name, y_name)        # [n, 7]
    extra_edges = gd.get_extra_edge_features()                 # [n, n, 3]
    edge_seqs   = gd.edge_seqs                                 # [n, n, 8, N_SUB]

    item = dict(
        edge_seqs   = torch.from_numpy(edge_seqs),
        extra_edges = torch.from_numpy(extra_edges),
        node_feats  = torch.from_numpy(node_feats),
        node_types  = torch.from_numpy(node_types),
        mask        = torch.ones(n, dtype=torch.bool),
        cols        = gd.cols,
        x_name      = x_name,
        y_name      = y_name,
    )

    if with_labels:
        labels_dict = gd.labels_by_xy.get((x_name, y_name), {})
        node_labels = np.full(n, -1, dtype=np.int64)  # -1 = ignored
        for v_name, cls_idx in labels_dict.items():
            if v_name in gd.cols:
                node_labels[gd.cols.index(v_name)] = cls_idx
        item["node_labels"] = torch.from_numpy(node_labels)

    return item


# ─── Collate ──────────────────────────────────────────────────────────────────

def collate_graph_fn(batch: list[dict]) -> dict:
    """
    Pads all tensors to the maximum n in the batch (or MAX_N, whichever is smaller).
    """
    max_n = min(max(item["edge_seqs"].shape[0] for item in batch), MAX_N)
    B = len(batch)
    C, L = N_CHANNELS, N_SUB
    E = N_EXTRA_EDGE
    F = N_NODE_FEATS

    edge_seqs   = torch.zeros(B, max_n, max_n, C, L)
    extra_edges = torch.zeros(B, max_n, max_n, E)
    node_feats  = torch.zeros(B, max_n, F)
    node_types  = torch.full((B, max_n), 2, dtype=torch.long)   # default = "other"
    mask        = torch.zeros(B, max_n, dtype=torch.bool)
    node_labels = torch.full((B, max_n), -1, dtype=torch.long)
    has_labels  = "node_labels" in batch[0]
    cols_list   = []
    xy_list     = []

    for b, item in enumerate(batch):
        n = min(item["edge_seqs"].shape[0], max_n)
        edge_seqs[b, :n, :n]   = item["edge_seqs"][:n, :n]
        extra_edges[b, :n, :n] = item["extra_edges"][:n, :n]
        node_feats[b, :n]      = item["node_feats"][:n]
        node_types[b, :n]      = item["node_types"][:n]
        mask[b, :n]            = True
        cols_list.append(item["cols"])
        xy_list.append((item["x_name"], item["y_name"]))
        if has_labels:
            node_labels[b, :n] = item["node_labels"][:n]

    out = dict(
        edge_seqs=edge_seqs, extra_edges=extra_edges,
        node_feats=node_feats, node_types=node_types,
        mask=mask, cols=cols_list, xy=xy_list,
    )
    if has_labels:
        out["node_labels"] = node_labels
    return out


# ─── Lightning Wrapper ────────────────────────────────────────────────────────

class V12Wrapper(pl.LightningModule):
    def __init__(
        self,
        d_edge: int   = D_EDGE,
        d_model: int  = D_MODEL,
        n_heads: int  = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        lr: float      = LR,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_v12_model(
            d_edge=d_edge, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.register_buffer("class_weights", CLASS_WEIGHTS)
        self.lr = lr

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(
            batch["edge_seqs"].to(self.device),
            batch["extra_edges"].to(self.device),
            batch["node_feats"].to(self.device),
            batch["node_types"].to(self.device),
            batch["mask"].to(self.device),
        )  # [B, n, 8]

    def _compute_loss(self, logits: torch.Tensor, batch: dict) -> torch.Tensor:
        labels = batch["node_labels"].to(self.device)  # [B, n], -1=ignore
        B, n, _ = logits.shape
        logits_flat = logits.view(B * n, N_CLASSES)
        labels_flat = labels.view(B * n)
        loss = F.cross_entropy(
            logits_flat, labels_flat,
            weight=self.class_weights.to(self.device),
            ignore_index=-1,
        )
        return loss

    def _compute_acc(self, logits: torch.Tensor, batch: dict) -> float:
        labels = batch["node_labels"].to(self.device)
        preds  = logits.argmax(dim=-1)  # [B, n]
        valid  = labels != -1
        if valid.sum() == 0:
            return 0.0
        return float((preds[valid] == labels[valid]).float().mean().item())

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss   = self._compute_loss(logits, batch)
        acc    = self._compute_acc(logits, batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc",  acc,  prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch)
        loss   = self._compute_loss(logits, batch)
        acc    = self._compute_acc(logits, batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc",  acc,  prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
        return [opt], [sched]


# ─── Train ────────────────────────────────────────────────────────────────────

def train(model_directory_path: str = "resources") -> None:
    os.makedirs(model_directory_path, exist_ok=True)

    # Build dataset if not cached
    shard_paths = sorted(glob.glob(os.path.join(CACHE_DIR, "v12_shard_*.pkl")))
    if not shard_paths:
        print("No cached shards found. Building dataset...")
        shard_paths = build_from_pickles(
            "data/X_train.pickle", "data/y_train.pickle",
            output_dir=CACHE_DIR, n_workers=48,
        )

    # Split shards into train/val (last shard = val)
    random.shuffle(shard_paths)
    val_shards   = shard_paths[:1]
    train_shards = shard_paths[1:]

    train_ds = GraphDataset(train_shards, shuffle_items=True)
    val_ds   = GraphDataset(val_shards,   shuffle_items=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_graph_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_graph_fn,
    )

    wrapper = V12Wrapper()
    n_params = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)
    print(f"V12Model: {n_params:,} parameters")

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=model_directory_path,
        filename=f"{MODEL_TAG}-best",
        monitor="val_acc", mode="max", save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="gpu", devices=2, strategy="ddp",
        precision="16-mixed",
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer.fit(wrapper, train_loader, val_loader)

    # Save final model state dict (strip DDP module. prefix)
    ckpt_path = checkpoint_cb.best_model_path
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    clean = {k.replace("model.", ""): v for k, v in state.items()
             if k.startswith("model.")}
    save_path = os.path.join(model_directory_path, f"{MODEL_TAG}.pt")
    torch.save(clean, save_path)
    print(f"Saved model to {save_path}")


# ─── Inference helpers ────────────────────────────────────────────────────────

def _load_model(model_directory_path: str) -> V12Model:
    path = os.path.join(model_directory_path, f"{MODEL_TAG}.pt")
    model = build_v12_model(
        d_edge=D_EDGE, d_model=D_MODEL,
        n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def _infer_one_graph(gd: GraphData, x_name: str, y_name: str, model: V12Model, device: str) -> dict[str, int]:
    """Return {v_name: predicted_class} for non-X,Y nodes."""
    item = _build_item(gd, x_name, y_name, with_labels=False)
    batch = collate_graph_fn([item])

    model.eval()
    with torch.no_grad():
        logits = model(
            batch["edge_seqs"].to(device),
            batch["extra_edges"].to(device),
            batch["node_feats"].to(device),
            batch["node_types"].to(device),
            batch["mask"].to(device),
        )  # [1, n, 8]

    preds = logits[0].argmax(dim=-1).cpu().numpy()  # [n]
    result: dict[str, int] = {}
    for v_idx, v_name in enumerate(gd.cols):
        if v_name in (x_name, y_name):
            continue
        result[v_name] = int(preds[v_idx])
    return result


def infer_batch_local(
    dfs: list[pd.DataFrame],
    model: V12Model,
    device: str = "cuda",
    cache_dir: str | None = None,
) -> list[pd.DataFrame]:
    """
    Local inference (for evaluate.py usage).
    Returns list of adjacency DataFrames.
    """
    from v12_features import _build_one_graph, GraphData
    from concurrent.futures import ProcessPoolExecutor

    results: list[pd.DataFrame] = []
    model.to(device).eval()

    for df in tqdm(dfs, desc="Infer (local)"):
        gd = _build_one_graph((df, None))
        if gd is None:
            cols = list(df.columns)
            adj = pd.DataFrame(0, index=cols, columns=cols)
            adj.loc["X", "Y"] = 1
            results.append(adj)
            continue

        preds = _infer_one_graph(gd, "X", "Y", model, device)

        cols = list(df.columns)
        adj = pd.DataFrame(0, index=cols, columns=cols)
        adj.loc["X", "Y"] = 1

        # Convert class idx to adjacency edges
        # (We only have node-level class predictions; reconstruct expected edges)
        # The submission format expects the node classification, not the adjacency —
        # so we just return the class index per node in the results.
        # For local eval against y_test_reduced.pickle we need the class directly.
        results.append((df.columns.tolist(), preds))

    return results


# ─── CrunchDAO infer ──────────────────────────────────────────────────────────

def infer(
    X_test: typing.Dict[str, pd.DataFrame],
    model_directory_path: str,
    id_column_name: str,
    prediction_column_name: str,
) -> pd.DataFrame:
    """CrunchDAO submission entry point."""
    from v12_features import _build_one_graph

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(model_directory_path).to(device).eval()

    submission: dict[str, int] = {}
    names = list(X_test.keys())

    for name in tqdm(names, desc="Inference"):
        df   = X_test[name]
        cols = list(df.columns)

        # Feature extraction (no cache on cloud)
        cache_path = None if IS_CLOUD_SUBMIT else None  # could add local cache if desired
        gd = _build_one_graph((df, None))
        if gd is None:
            # Fallback: predict Independent for all
            for v in cols:
                if v not in ("X", "Y"):
                    key = f"{name}_{v}"
                    submission[key] = CLASS_NAMES.index("Independent")
            continue

        preds = _infer_one_graph(gd, "X", "Y", model, device)
        for v_name, cls_idx in preds.items():
            key = f"{name}_{v_name}"
            submission[key] = cls_idx

    s = pd.Series(submission).reset_index()
    s.columns = [id_column_name, prediction_column_name]
    return s


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        # DDP non-rank-0 processes: just run train, evaluation on rank 0 only
        train()
    else:
        train()
        # Optional: run local evaluation after training
        # from evaluate import evaluate_local
        # evaluate_local(...)
