"""
Utilities for loading the DGraph-Fin / DGraph-Fin2 dataset into DGL.

We rely on the **official DGraph-Fin release** (`dgraphfin.npz`) for:

- node features `x` (shape `[num_nodes, num_features]`)
- multi-class labels `y` in `{0, 1, 2, 3}` (fraud = 1, normal = 0, background = 2/3)
- edge list `edge_index` (shape `[num_edges, 2]`)
- per-edge timestamps `edge_timestamp`
- train/valid/test splits as **lists of node indices**

From this, we construct a DGLGraph with:

- `g.ndata['feat']`   : float tensor of shape `[N, F]`
- `g.ndata['label']`  : **binary** labels in `{0, 1}` where `1` = fraud (class 1), `0` = all others
- `g.ndata['train_mask']`, `g.ndata['val_mask']`, `g.ndata['test_mask']` : boolean masks

If available, we additionally incorporate **DGraph-Fin2 temporal information**
from `data/DGraphFin2/raw/`:

- `dgraphfinv2_edge_timestamp.npy`  → `g.edata['timestamp']` (overrides OG timestamps)
- `dgraphfinv2_node_timestamp.npy`  → `g.ndata['timestamp']`
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import dgl
import numpy as np
import torch


@dataclass
class DGraphFinPaths:
    """
    Container for dataset file paths rooted at the directory passed to
    `DGraphFinDataset`.

    By default we expect:

    - OG data under:   `<root>/../DGraphFin/dgraphfin.npz`
    - v2 timestamps under: `<root>/raw/dgraphfinv2_*_timestamp.npy`
    """

    # Typically something like "data/DGraphFin2"
    root: str

    # Relative location of the OG npz file, interpreted from `root`.
    og_dir: str = "DGraphFin"
    og_file: str = "dgraphfin.npz"

    # Optional v2 timestamp files (present in DGraph-Fin2 archive).
    v2_edge_ts_file: str = "dgraphfinv2_edge_timestamp.npy"
    v2_node_ts_file: str = "dgraphfinv2_node_timestamp.npy"

    def resolve(self, filename: Optional[str]) -> Optional[str]:
        if filename is None:
            return None
        return os.path.join(self.root, filename)


class DGraphFinDataset:
    """
    Thin wrapper around a single DGLGraph for backwards-compatible access.

    This mirrors the original dataset's `__len__`/`__getitem__` API while
    internally storing a DGL graph instead of a PyG `Data` object.
    """

    def __init__(self, graph: dgl.DGLGraph) -> None:
        self.graph = graph

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        if idx != 0:
            raise IndexError("DGraphFin dataset contains a single graph at index 0.")
        return self.graph


    def _create_random_masks(
        num_nodes: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fallback: create random train/val/test masks if not provided.
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(num_nodes)

        n_train = int(train_ratio * num_nodes)
        n_val = int(val_ratio * num_nodes)

        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask


def load_dgraphfin_graph(
    root: str,
    paths: Optional[DGraphFinPaths] = None,
) -> dgl.DGLGraph:
    """
    Load DGraph-Fin / DGraph-Fin2 into a single DGLGraph with node/edge data.

    The resulting graph has:

    - g.ndata['feat']        : node features (float32) of shape [N, F]
    - g.ndata['label']       : binary labels in {0, 1}
    - g.ndata['train_mask']  : boolean mask for training nodes
    - g.ndata['val_mask']    : boolean mask for validation nodes
    - g.ndata['test_mask']   : boolean mask for test nodes
    - g.edata['timestamp']   : edge timestamps (int64, optional)
    - g.ndata['timestamp']   : node timestamps (int64, optional)
    """
    p = paths or DGraphFinPaths(root=root)

    # `root` is typically `data/DGraphFin2`; OG file lives in a sibling dir.
    data_root = os.path.abspath(os.path.join(p.root, os.pardir))
    og_path = os.path.join(data_root, p.og_dir, p.og_file)

    if not os.path.exists(og_path):
        raise FileNotFoundError(
            f"Expected DGraph-Fin npz at '{og_path}', but it was not found. "
            "Make sure you have unpacked DGraphFin.zip into the data directory."
        )

    with np.load(og_path) as f:
        x_np = f["x"]  # [N, F]
        y_multiclass = f["y"]  # [N], values in {0,1,2,3}
        edge_index_np = f["edge_index"]  # [E, 2]

        # Convert to tensors.
        x = torch.from_numpy(x_np).float()
        # Binary labels: fraud (class 1) vs all others.
        y = torch.from_numpy((y_multiclass == 1).astype(np.int64)).long()

        num_nodes = x.shape[0]

        # Edge list for DGL: separate source and destination arrays.
        src = torch.from_numpy(edge_index_np[:, 0]).long()
        dst = torch.from_numpy(edge_index_np[:, 1]).long()

        g = dgl.graph((src, dst), num_nodes=num_nodes)

        # Node features / labels.
        g.ndata["feat"] = x
        g.ndata["label"] = y

        # OG edge timestamps (if present in the npz).
        if "edge_timestamp" in f:
            edge_ts = torch.from_numpy(f["edge_timestamp"]).long()
            if edge_ts.shape[0] == g.num_edges():
                g.edata["timestamp"] = edge_ts

        # Train/valid/test splits are provided as node index lists.
        if all(k in f for k in ("train_mask", "valid_mask", "test_mask")):
            train_nodes = torch.from_numpy(f["train_mask"]).long()
            val_nodes = torch.from_numpy(f["valid_mask"]).long()
            test_nodes = torch.from_numpy(f["test_mask"]).long()

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[train_nodes] = True
            val_mask[val_nodes] = True
            test_mask[test_nodes] = True
        else:
            train_mask, val_mask, test_mask = _create_random_masks(num_nodes)

        g.ndata["train_mask"] = train_mask
        g.ndata["val_mask"] = val_mask
        g.ndata["test_mask"] = test_mask

    # ------------------------------------------------------------------ #
    # 2) Augment with DGraph-Fin2 temporal information if available
    # ------------------------------------------------------------------ #
    v2_raw_dir = os.path.join(p.root, "raw")

    edge_ts_v2_path = os.path.join(v2_raw_dir, p.v2_edge_ts_file)
    if os.path.exists(edge_ts_v2_path):
        edge_ts_v2 = np.load(edge_ts_v2_path)
        if edge_ts_v2.shape[0] == g.num_edges():
            g.edata["timestamp"] = torch.from_numpy(edge_ts_v2).long()

    node_ts_v2_path = os.path.join(v2_raw_dir, p.v2_node_ts_file)
    if os.path.exists(node_ts_v2_path):
        node_ts_v2 = np.load(node_ts_v2_path)
        if node_ts_v2.shape[0] == num_nodes:
            g.ndata["timestamp"] = torch.from_numpy(node_ts_v2).long()

    return g


def load_dgraphfin_dataset(
    root: str,
    transform=None,
    pre_transform=None,
    paths: Optional[DGraphFinPaths] = None,
) -> DGraphFinDataset:
    """
    Convenience function to load the DGraph-Fin dataset from a given root.

    Example
    -------
    >>> dataset = load_dgraphfin_dataset(root="data/DGraphFin2")
    >>> g = dataset[0]
    >>> print(g)
    """
    g = load_dgraphfin_graph(root=root, paths=paths)
    return DGraphFinDataset(g)


