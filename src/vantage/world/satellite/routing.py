"""Shortest-path routing on ISL graphs.

Computes all-pairs shortest path delay matrix and predecessor matrix
for on-demand path reconstruction. Default backend uses networkx Dijkstra.

The RoutingComputer Protocol allows plugging in alternative backends
(e.g., native C++ Dijkstra, GPU-accelerated Floyd-Warshall).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from vantage.domain import ISLGraph


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """All-pairs routing result: delay matrix + predecessor matrix.

    delay_matrix: (n, n) one-way ISL delay in ms.
    predecessor_matrix: (n, n) int32, predecessor of dst on shortest path from src. -1 = no path.
    """

    shell_id: int
    timeslot: int
    num_sats: int
    delay_matrix: NDArray[np.float64]
    predecessor_matrix: NDArray[np.int32]

    def __post_init__(self) -> None:
        expected = (self.num_sats, self.num_sats)
        if self.delay_matrix.shape != expected:
            raise ValueError(
                f"delay_matrix shape {self.delay_matrix.shape} != expected {expected}"
            )
        if self.delay_matrix.dtype != np.float64:
            raise ValueError(
                f"delay_matrix dtype must be float64, got {self.delay_matrix.dtype}"
            )
        if self.predecessor_matrix.shape != expected:
            raise ValueError(
                f"predecessor_matrix shape {self.predecessor_matrix.shape} != expected {expected}"
            )
        if self.predecessor_matrix.dtype != np.int32:
            raise ValueError(
                f"predecessor_matrix dtype must be int32, got {self.predecessor_matrix.dtype}"
            )


class RoutingComputer(Protocol):
    """Protocol for all-pairs shortest path computation.

    Default implementation: ``compute_all_pairs`` (networkx Dijkstra).
    Alternative backends can implement this to accelerate routing.
    """

    def __call__(self, graph: ISLGraph) -> RoutingResult: ...


def compute_all_pairs(graph: ISLGraph) -> RoutingResult:
    """Compute all-pairs shortest paths via Dijkstra.

    Returns delay_matrix and predecessor_matrix for on-demand
    path reconstruction.
    """
    n = graph.num_sats
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for edge in graph.edges:
        g.add_edge(edge.sat_a, edge.sat_b, weight=edge.delay)

    delay_matrix = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(delay_matrix, 0.0)
    predecessor_matrix = np.full((n, n), -1, dtype=np.int32)
    np.fill_diagonal(predecessor_matrix, np.arange(n, dtype=np.int32))

    for src, (lengths, paths) in nx.all_pairs_dijkstra(g, weight="weight"):
        for tgt, length in lengths.items():
            delay_matrix[src, tgt] = length
        for tgt, path in paths.items():
            if len(path) >= 2:
                predecessor_matrix[src, tgt] = path[-2]

    delay_matrix.flags.writeable = False
    predecessor_matrix.flags.writeable = False

    return RoutingResult(
        shell_id=graph.shell_id,
        timeslot=graph.timeslot,
        num_sats=n,
        delay_matrix=delay_matrix,
        predecessor_matrix=predecessor_matrix,
    )
