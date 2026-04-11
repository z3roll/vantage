"""Shortest-path routing on ISL graphs.

Computes all-pairs shortest path delay matrix and predecessor matrix
for on-demand path reconstruction. Default backend uses networkx Dijkstra.

The RoutingComputer Protocol allows plugging in alternative backends
(e.g., native C++ Dijkstra, GPU-accelerated Floyd-Warshall).

The :func:`first_hop_on_path` helper reconstructs the first ISL hop
of a shortest path from ``src`` to ``dst`` by walking *backwards*
through the predecessor matrix. This is the primitive used by the
FIB builder in :mod:`vantage.control.policy.common.fib_builder` to
turn an all-pairs routing result into per-satellite forwarding
entries.
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


def first_hop_on_path(
    predecessor_matrix: NDArray[np.int32],
    src: int,
    dst: int,
) -> int:
    """Return the first ISL hop on the shortest path from ``src`` to ``dst``.

    ``predecessor_matrix[s, t]`` stores the *predecessor of t* on the
    shortest path from ``s`` — i.e., the second-to-last node. To find
    the *first* hop out of ``src``, we walk backwards from ``dst``:
    keep stepping to ``predecessor_matrix[src, current]`` until that
    predecessor equals ``src`` itself, at which point ``current`` is
    the first hop we were looking for.

    Special cases:

    * ``src == dst``  → returns ``src`` (the path has length 0, no hop).
    * ``predecessor == -1`` at any step → returns ``-1`` (unreachable).
    * ``src`` is its own predecessor in the diagonal entry (by the
      current :func:`compute_all_pairs` contract), which terminates
      the walk correctly on the one-hop case.

    Complexity: O(path length), dominated by number of ISL hops.
    For Starlink-scale routes (≤10 hops end-to-end) this is trivial
    and does not warrant vectorization.

    Args:
        predecessor_matrix: All-pairs predecessor matrix (int32) from
            a :class:`RoutingResult`. Shape ``(n_sats, n_sats)``.
        src: Source satellite id (row index).
        dst: Destination satellite id (column index).

    Returns:
        The id of the next-hop satellite out of ``src`` toward ``dst``,
        ``src`` itself if ``src == dst``, or ``-1`` if unreachable.
    """
    if src == dst:
        return src
    current = dst
    # Walk backwards through predecessors until the predecessor equals src.
    # Bound the loop by the matrix dimension so a malformed predecessor
    # matrix can't spin forever.
    max_hops = predecessor_matrix.shape[0]
    for _ in range(max_hops):
        prev = int(predecessor_matrix[src, current])
        if prev < 0:
            return -1
        if prev == src:
            return current
        current = prev
    # Malformed matrix — a cycle that doesn't terminate at src.
    raise ValueError(
        f"first_hop_on_path did not terminate for src={src}, dst={dst}; "
        f"predecessor matrix may contain a cycle"
    )


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
