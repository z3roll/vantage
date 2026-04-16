"""GroundOnly controller: ground-delay oracle baseline.

Uses measured ground delays (from traceroute data) to select PoPs.
Pairs without measurements fall back to nearest-PoP baseline.
"""

from __future__ import annotations

import logging

from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_routing_plane_with_overrides,
    compute_cell_sat_cost,
    compute_e2e_overrides,
)
from vantage.domain import CellGrid, Endpoint, NetworkSnapshot, RoutingPlane
from vantage.world.ground import GroundDelay, MeasuredGroundDelay

_log = logging.getLogger(__name__)


class GroundOnlyController:
    """Baseline: route to the PoP with lowest *measured* ground delay.

    Only ``(pop, dest)`` pairs present in the supplied
    :class:`GroundDelay` table are considered. Unknown pairs fall back
    to nearest-PoP baseline.
    """

    def __init__(
        self,
        ground_delay: GroundDelay | None = None,
        endpoints: dict[str, Endpoint] | None = None,
    ) -> None:
        self._endpoints = endpoints or {}
        self._ground_delay: GroundDelay = ground_delay or MeasuredGroundDelay.empty()
        if (
            isinstance(self._ground_delay, MeasuredGroundDelay)
            and len(self._ground_delay) == 0
        ):
            _log.warning(
                "GroundOnlyController: ground_delay table is empty; "
                "will degrade to nearest-PoP baseline."
            )

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        version: int = 0,
    ) -> RoutingPlane:
        dest_names = [
            ep.name
            for ep in self._endpoints.values()
            if not ep.name.startswith("terminal_")
        ]

        def ground_cost_fn(pop_code: str, dest: str) -> float | None:
            try:
                return self._ground_delay.estimate(pop_code, dest) * 2
            except KeyError:
                return None

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
            ground_cost_fn=ground_cost_fn,
            dest_names=dest_names,
        )
        return build_routing_plane_with_overrides(
            snapshot, cell_grid, overrides,
            baseline=baseline, version=version,
        )
