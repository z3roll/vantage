"""NearestPoP controller: hot-potato baseline.

Routes every cell to its geographically-nearest PoP. The satellite
path is represented by the refresh-time :class:`SatPathTable`
(Dijkstra output) and per-PoP egress candidates are precomputed into
a :class:`PopEgressTable`; both artifacts travel in the emitted
:class:`RoutingPlane` so the data plane never recomputes routing.
"""

from __future__ import annotations

import time
from types import MappingProxyType
from collections.abc import Mapping

from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_pop_egress_table,
    build_sat_path_table,
)
from vantage.domain import CellGrid, NetworkSnapshot, RoutingPlane


class NearestPoPController:
    """Baseline: route to the PoP with the lowest satellite-segment cost."""

    def __init__(self) -> None:
        # Per-step wall-clock timings (ms) for the most recent
        # ``compute_routing_plane`` invocation. ``run.py`` reads this
        # after a refresh to export per-step breakdowns to the
        # dashboard. Empty until the first call.
        self._last_timing: Mapping[str, float] = MappingProxyType({})

    @property
    def last_timing(self) -> Mapping[str, float]:
        """Step timings (ms) from the most recent plan build."""
        return self._last_timing

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        version: int = 0,
    ) -> RoutingPlane:
        perf = time.perf_counter
        t0 = perf()
        cell_to_pop = build_cell_to_pop_nearest(
            cell_grid=cell_grid,
            pops=snapshot.infra.pops,
            built_at=snapshot.time_s,
            version=version,
        )
        t1 = perf()
        sat_paths = build_sat_path_table(snapshot, version=version)
        t2 = perf()
        pop_egress = build_pop_egress_table(snapshot, version=version)
        t3 = perf()

        self._last_timing = MappingProxyType({
            "cell_to_pop_ms": (t1 - t0) * 1000.0,
            "sat_paths_ms": (t2 - t1) * 1000.0,
            "pop_egress_ms": (t3 - t2) * 1000.0,
        })

        return RoutingPlane(
            cell_to_pop=cell_to_pop,
            sat_paths=sat_paths,
            pop_egress=pop_egress,
            version=version,
            built_at=snapshot.time_s,
        )
