"""NearestPoP controller: hot-potato baseline.

Routes every cell to its geographically-nearest PoP.
The satellite path is determined by per-sat FIBs derived from Dijkstra.
"""

from __future__ import annotations

from vantage.control.policy.common.fib_builder import (
    build_routing_plane_nearest_pop,
)
from vantage.domain import CellGrid, NetworkSnapshot, RoutingPlane


class NearestPoPController:
    """Baseline: route to the PoP with the lowest satellite-segment cost."""

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        version: int = 0,
    ) -> RoutingPlane:
        return build_routing_plane_nearest_pop(
            snapshot=snapshot,
            cell_grid=cell_grid,
            version=version,
        )
