"""
Utilities for loading the DGraph-Fin / DGraph-Fin2 dataset into PyTorch Geometric.

We rely on the **official DGraph-Fin release** (`dgraphfin.npz`) for:

- node features `x` (shape `[num_nodes, num_features]`)
- multi-class labels `y` in `{0, 1, 2, 3}` (fraud = 1, normal = 0, background = 2/3)
- edge list `edge_index` (shape `[num_edges, 2]`)
- per-edge timestamps `edge_timestamp`
- train/valid/test splits as **lists of node indices**

From this, we construct a PyG `Data` object with:

- `x`: float tensor of shape `[N, F]`
- `y`: **binary** labels in `{0, 1}` where `1` = fraud (class 1), `0` = all others
- `edge_index`: long tensor of shape `[2, E]`
- boolean `train_mask`, `val_mask`, `test_mask`

If available, we additionally incorporate **DGraph-Fin2 temporal information**
from `data/DGraphFin2/raw/`:

- `dgraphfinv2_edge_timestamp.npy`  → `data.edge_timestamp` (overrides OG timestamps)
- `dgraphfinv2_node_timestamp.npy`  → `data.node_timestamp`

This way, the loader uses the trustworthy OG topology/labels while enriching
the graph with v2-style temporal node/edge information when present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import index_to_mask


@dataclass
class DGraphFinPaths:
    """
    Container for dataset file paths rooted at the directory passed to
    `DGraphFinDataset`.

    By default we expect:

    - OG data under:   `<root>/../DGraphFin/dgraphfin.npz`
    - v2 timestamps under: `<root>/raw/dgraphfinv2_*_timestamp.npy`

    You can adjust these names if you have a slightly different layout.
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


class DGraphFinDataset(InMemoryDataset):
    """
    DGraph-Fin / DGraph-Fin2 dataset loader for PyTorch Geometric.

    This loader assumes that the official `dgraphfin.npz` has been placed
    under a sibling directory of `root` (by default, `data/DGraphFin`) and
    that DGraph-Fin2 timestamp arrays are unpacked under `root/raw`:

    - `data/DGraphFin/dgraphfin.npz`
    - `data/DGraphFin2/raw/dgraphfinv2_edge_timestamp.npy` (optional)
    - `data/DGraphFin2/raw/dgraphfinv2_node_timestamp.npy` (optional)

    The resulting `Data` object has binary labels suitable for
    `BCEWithLogitsLoss` and boolean train/val/test masks.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        paths: Optional[DGraphFinPaths] = None,
    ) -> None:
        self.paths = paths or DGraphFinPaths(root=root)
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

        # Torch 2.6+ defaults `weights_only=True`, which is unsuitable for
        # arbitrary `Data` objects. Explicitly disable it here.
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        # Not used directly by our loader, but keep a meaningful placeholder.
        return [self.paths.og_file]

    @property
    def processed_file_names(self) -> list[str]:
        # Single collated Data object saved to disk.
        return ["dgraphfin.pt"]

    def download(self) -> None:  # pragma: no cover - offline, user-managed
        # The dataset is already downloaded by the user (per project instructions).
        # If needed, you can implement extraction logic here.
        pass

    def process(self) -> None:
        data = self._load_raw_data()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(self.collate([data]), self.processed_paths[0])

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _load_raw_data(self) -> Data:
        """
        Load numpy arrays from disk and construct a `torch_geometric.data.Data`.

        Primary source is the official `dgraphfin.npz` file (OG DGraph-Fin),
        with optional augmentation from DGraph-Fin2 timestamp arrays.
        """
        # ------------------------------------------------------------------ #
        # 1) Load OG DGraph-Fin from `dgraphfin.npz`
        # ------------------------------------------------------------------ #
        p = self.paths

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

            # Transpose edge_index to shape [2, E] for PyG.
            edge_index = torch.from_numpy(edge_index_np).long().t().contiguous()

            data = Data(x=x, edge_index=edge_index, y=y)

            num_nodes = x.size(0)

            # OG edge timestamps (if present in the npz).
            if "edge_timestamp" in f:
                edge_ts = torch.from_numpy(f["edge_timestamp"]).long()
                # Expect one timestamp per edge.
                if edge_ts.numel() == edge_index.size(1):
                    data.edge_timestamp = edge_ts

            # Train/valid/test splits are provided as node index lists.
            if all(k in f for k in ("train_mask", "valid_mask", "test_mask")):
                train_nodes = torch.from_numpy(f["train_mask"]).long()
                val_nodes = torch.from_numpy(f["valid_mask"]).long()
                test_nodes = torch.from_numpy(f["test_mask"]).long()

                data.train_mask = index_to_mask(train_nodes, size=num_nodes)
                data.val_mask = index_to_mask(val_nodes, size=num_nodes)
                data.test_mask = index_to_mask(test_nodes, size=num_nodes)
            else:
                train_mask, val_mask, test_mask = self._create_random_masks(num_nodes)
                data.train_mask = train_mask
                data.val_mask = val_mask
                data.test_mask = test_mask

        # ------------------------------------------------------------------ #
        # 2) Augment with DGraph-Fin2 temporal information if available
        # ------------------------------------------------------------------ #
        v2_raw_dir = os.path.join(p.root, "raw")

        edge_ts_v2_path = os.path.join(v2_raw_dir, p.v2_edge_ts_file)
        if os.path.exists(edge_ts_v2_path):
            edge_ts_v2 = np.load(edge_ts_v2_path)
            if edge_ts_v2.shape[0] == data.edge_index.size(1):
                data.edge_timestamp = torch.from_numpy(edge_ts_v2).long()

        node_ts_v2_path = os.path.join(v2_raw_dir, p.v2_node_ts_file)
        if os.path.exists(node_ts_v2_path):
            node_ts_v2 = np.load(node_ts_v2_path)
            if node_ts_v2.shape[0] == num_nodes:
                data.node_timestamp = torch.from_numpy(node_ts_v2).long()

        return data

    @staticmethod
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
    >>> dataset = load_dgraphfin_dataset(root=\"data/DGraphFin\")
    >>> data = dataset[0]
    >>> print(data)
    """
    return DGraphFinDataset(root=root, transform=transform, pre_transform=pre_transform, paths=paths)


