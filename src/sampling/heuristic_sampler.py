"""
Heuristic biased neighbor sampler for imbalanced graphs.

Idea:
- For seed nodes with positive labels (anomalies), sample a larger number of
  neighbors (K_pos) to capture rich local context.
- For seed nodes with negative labels (normal), sample fewer neighbors (Kneg)
  to reduce computation on the majority class.

This implementation builds small subgraphs around seed nodes using Python-level
sampling logic and returns `torch_geometric.data.Data` objects similar to those
emitted by `NeighborLoader`. It is intentionally simple and geared towards
research experimentation rather than maximum speed.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


class HeuristicNeighborLoader:
    """
    Mini-batch loader that performs heuristic neighbor sampling.

    Parameters
    ----------
    data : Data
        Full graph data object.
    input_nodes : Tensor
        Indices of nodes to use as training seeds (e.g., train_mask nonzeros).
    batch_size : int
        Number of seed nodes per batch.
    num_hops : int
        Number of sampling hops (graph layers).
    k_pos : int
        Number of neighbors to sample for positive-labeled nodes.
    k_neg : int
        Number of neighbors to sample for negative-labeled nodes.
    shuffle : bool
        Whether to shuffle the input node order each epoch.
    """

    def __init__(
        self,
        data: Data,
        input_nodes: torch.Tensor,
        batch_size: int,
        num_hops: int = 2,
        k_pos: int = 25,
        k_neg: int = 5,
        shuffle: bool = True,
    ) -> None:
        self.data = data
        self.input_nodes = input_nodes.clone().detach().cpu()
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.shuffle = shuffle

        self._neighbors = self._build_neighbors(data.edge_index, data.num_nodes)

    @staticmethod
    def _build_neighbors(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
        """
        Build an undirected adjacency list from edge_index.
        """
        src, dst = edge_index
        neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()
        for s, d in zip(src_np, dst_np):
            if d not in neighbors[s]:
                neighbors[s].append(d)
            if s not in neighbors[d]:
                neighbors[d].append(s)
        return neighbors

    def __len__(self) -> int:
        return (self.input_nodes.numel() + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterable[Data]:
        indices = self.input_nodes.clone()
        if self.shuffle:
            perm = torch.randperm(indices.numel())
            indices = indices[perm]

        for start in range(0, indices.numel(), self.batch_size):
            seeds = indices[start : start + self.batch_size]
            if seeds.numel() == 0:
                continue
            yield self._build_batch(seeds)

    def _sample_neighbors_for_node(self, node: int, label: int) -> List[int]:
        neighbors = self._neighbors[node]
        if not neighbors:
            return []
        k = self.k_pos if label == 1 else self.k_neg
        if k <= 0:
            return []
        if len(neighbors) <= k:
            return neighbors
        return random.sample(neighbors, k)

    def _build_batch(self, seeds: torch.Tensor) -> Data:
        data = self.data
        seeds_list = seeds.tolist()

        # Start from seeds and grow the node set for num_hops.
        node_set = set(seeds_list)
        frontier = seeds_list

        for _ in range(self.num_hops):
            new_frontier: List[int] = []
            for nid in frontier:
                label = int(data.y[nid].item())
                sampled_neighbors = self._sample_neighbors_for_node(nid, label)
                for nb in sampled_neighbors:
                    if nb not in node_set:
                        node_set.add(nb)
                        new_frontier.append(nb)
            frontier = new_frontier
            if not frontier:
                break

        # Build subgraph.
        node_idx = torch.tensor(sorted(node_set), dtype=torch.long)
        sub_edge_index, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)
        x_sub = data.x[node_idx]
        y_sub = data.y[node_idx]

        # Map seeds to local indices.
        global_to_local = {nid.item(): i for i, nid in enumerate(node_idx)}
        seed_local_idx = torch.tensor(
            [global_to_local[nid] for nid in seeds_list], dtype=torch.long
        )

        batch = Data(x=x_sub, edge_index=sub_edge_index, y=y_sub)
        batch.batch_size = seed_local_idx.numel()
        batch.seed_idx = seed_local_idx
        return batch


