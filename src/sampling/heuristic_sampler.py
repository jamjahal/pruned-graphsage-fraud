"""
Heuristic biased neighbor sampler for imbalanced graphs (DGL version).

Idea:
- Positive (fraud) seeds sample up to `k_pos` neighbors to capture richer context.
- Negative seeds sample up to `k_neg` neighbors to keep majority class cheap.

Returns DGL subgraphs plus the local indices of the seed nodes so the training
loop can compute losses only on the intended targets, similar to PyG's
NeighborLoader contract.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import dgl
import torch
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader


@dataclass
class SubgraphBatch:
    graph: dgl.DGLGraph
    seed_idx: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.seed_idx.numel()


def _build_neighbors(graph: dgl.DGLGraph) -> List[List[int]]:
    """
    Construct an undirected adjacency list from the (possibly directed) graph.
    """
    num_nodes = graph.num_nodes()
    neighbors: List[set[int]] = [set() for _ in range(num_nodes)]
    src, dst = graph.edges()
    src_list = src.tolist()
    dst_list = dst.tolist()
    for s, d in zip(src_list, dst_list):
        neighbors[s].add(d)
        neighbors[d].add(s)
    return [list(nbs) for nbs in neighbors]


class HeuristicNeighborLoader:
    """
    Mini-batch loader that performs label-aware neighbor sampling using DGL graphs.
    """

    def __init__(
        self,
        graph: dgl.DGLGraph,
        input_nodes: torch.Tensor,
        batch_size: int,
        num_hops: int = 2,
        k_pos: int = 25,
        k_neg: int = 5,
        shuffle: bool = True,
    ) -> None:
        self.graph = graph
        self.labels = graph.ndata["label"]
        self.input_nodes = input_nodes.clone().detach().cpu()
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.shuffle = shuffle

        self._neighbors = _build_neighbors(graph)

    def __len__(self) -> int:
        return (self.input_nodes.numel() + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterable[SubgraphBatch]:
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

    def _build_batch(self, seeds: torch.Tensor) -> SubgraphBatch:
        graph = self.graph
        seeds_list = seeds.tolist()

        node_set = set(seeds_list)
        frontier = seeds_list

        for _ in range(self.num_hops):
            new_frontier: List[int] = []
            for nid in frontier:
                label = int(self.labels[nid].item())
                sampled_neighbors = self._sample_neighbors_for_node(nid, label)
                for nb in sampled_neighbors:
                    if nb not in node_set:
                        node_set.add(nb)
                        new_frontier.append(nb)
            frontier = new_frontier
            if not frontier:
                break

        node_idx = torch.tensor(sorted(node_set), dtype=torch.long)
        subgraph = dgl.node_subgraph(graph, node_idx)

        global_to_local = {int(nid): i for i, nid in enumerate(node_idx.tolist())}
        seed_local_idx = torch.tensor(
            [global_to_local[nid] for nid in seeds_list], dtype=torch.long
        )

        return SubgraphBatch(graph=subgraph, seed_idx=seed_local_idx)


def build_uniform_neighbor_dataloader(
    graph: dgl.DGLGraph,
    input_nodes: torch.Tensor,
    num_neighbors: Sequence[int],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Construct a DGL NodeDataLoader for standard uniform neighbor sampling.

    Parameters
    ----------
    graph : dgl.DGLGraph
        Full training graph containing node features, labels, and masks.
    input_nodes : torch.Tensor
        Node IDs that should serve as seeds (e.g., train mask indices).
    num_neighbors : Sequence[int]
        Fan-out per hop, like PyG's `num_neighbors`.
    batch_size : int
        Number of seed nodes per mini-batch.
    shuffle : bool
        Whether to reshuffle the order of seeds every epoch.
    num_workers : int
        Number of sampler workers (0 = iterate in main process).
    """
    sampler = MultiLayerNeighborSampler(num_neighbors)
    dataloader = DataLoader(
        graph,
        input_nodes,
        sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
    )
    return dataloader


