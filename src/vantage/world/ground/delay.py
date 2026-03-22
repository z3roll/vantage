"""Ground delay estimation models (L2/L3).

Used when no observed data is available in the cache.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from vantage.common import C_FIBER_KM_S, DEFAULT_DETOUR_FACTOR, haversine_km


class GroundDelay(Protocol):
    """Protocol for ground segment delay estimation (PoP → destination)."""

    def estimate(
        self, pop_lat: float, pop_lon: float, dest_lat: float, dest_lon: float
    ) -> float:
        """Estimate one-way ground delay in ms."""
        ...


class HaversineDelay:
    """L2: Ground delay = haversine distance × detour_factor / fiber_speed."""

    def __init__(
        self, c_fiber_km_s: float = C_FIBER_KM_S, detour_factor: float = DEFAULT_DETOUR_FACTOR
    ) -> None:
        self._c_fiber = c_fiber_km_s
        self._detour = detour_factor

    def estimate(
        self, pop_lat: float, pop_lon: float, dest_lat: float, dest_lon: float
    ) -> float:
        """Returns one-way ground delay in ms."""
        dist_km = haversine_km(pop_lat, pop_lon, dest_lat, dest_lon)
        return dist_km * self._detour / self._c_fiber * 1000


class FiberGraphDelay:
    """L3: Ground delay via shortest path on ITU fiber network graph.

    Loads preprocessed fiber_graph.json. For a given (PoP, destination),
    snaps both to the nearest graph node, runs Dijkstra, and converts
    fiber distance to delay in ms.
    """

    def __init__(
        self,
        fiber_graph_path: str | Path,
        c_fiber_km_s: float = C_FIBER_KM_S,
    ) -> None:
        self._c_fiber = c_fiber_km_s
        self._load_graph(Path(fiber_graph_path))

    def _load_graph(self, path: Path) -> None:
        with path.open() as f:
            data = json.load(f)

        self._node_coords = np.array(
            [[n["lat_deg"], n["lon_deg"]] for n in data["nodes"]],
            dtype=np.float64,
        )
        num_nodes = len(data["nodes"])

        # Build adjacency list
        self._adj: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
        for edge in data["edges"]:
            a, b, d = edge["node_a"], edge["node_b"], edge["distance_km"]
            self._adj[a].append((b, d))
            self._adj[b].append((a, d))

    def _snap_to_node(self, lat: float, lon: float) -> int:
        """Find nearest graph node to a given coordinate."""
        from math import cos, radians

        dlat = self._node_coords[:, 0] - lat
        dlon = (self._node_coords[:, 1] - lon) * cos(radians(lat))
        dist_sq = dlat * dlat + dlon * dlon
        return int(np.argmin(dist_sq))

    def _dijkstra(self, src: int, dst: int) -> float:
        """Shortest path distance (km) between two nodes."""
        dist = [float("inf")] * len(self._adj)
        dist[src] = 0.0
        heap: list[tuple[float, int]] = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                return d
            if d > dist[u]:
                continue
            for v, w in self._adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        return float("inf")

    def estimate(
        self, pop_lat: float, pop_lon: float, dest_lat: float, dest_lon: float
    ) -> float:
        """Returns one-way ground delay in ms."""
        src_node = self._snap_to_node(pop_lat, pop_lon)
        dst_node = self._snap_to_node(dest_lat, dest_lon)

        if src_node == dst_node:
            dist_km = haversine_km(pop_lat, pop_lon, dest_lat, dest_lon)
        else:
            dist_km = self._dijkstra(src_node, dst_node)
            if dist_km == float("inf"):
                dist_km = haversine_km(pop_lat, pop_lon, dest_lat, dest_lon) * 1.5

        return dist_km / self._c_fiber * 1000
