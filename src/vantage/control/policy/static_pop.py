"""StaticPoP controller: best static PoP per destination from cache.

Uses sat_cost + ground_cost from GroundKnowledge cache only (no estimation).
"""

from __future__ import annotations

from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_routing_plane_with_overrides,
    compute_cell_sat_cost,
    compute_e2e_overrides,
)
from vantage.domain import CellGrid, NetworkSnapshot, RoutingPlane
from vantage.world.ground import GroundKnowledge


class StaticPoPController:
    """Baseline: static per-destination PoP from ground knowledge cache."""

    def __init__(
        self,
        ground_knowledge: GroundKnowledge | None = None,
    ) -> None:
        self._gk = ground_knowledge or GroundKnowledge()

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        version: int = 0,
    ) -> RoutingPlane:
        # Derive dest_names from cached entries
        dest_names = {dest for _, dest in self._gk.all_entries()}

        cell_sat_cost = compute_cell_sat_cost(snapshot, cell_grid)
        baseline = build_cell_to_pop_nearest(
            cell_grid=cell_grid,
            pops=snapshot.infra.pops,
            built_at=snapshot.time_s,
            version=version,
        )
        overrides = compute_e2e_overrides(
            cell_grid=cell_grid,
            pops=snapshot.infra.pops,
            baseline=baseline,
            cell_sat_cost=cell_sat_cost,
            ground_cost_fn=self._gk.get,
            dest_names=dest_names,
        )
        return build_routing_plane_with_overrides(
            snapshot, cell_grid, overrides,
            baseline=baseline, version=version,
        )
