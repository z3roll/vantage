"""NearestPoP controller: capacity-aware hot-potato baseline.

Default behaviour routes every cell to its geographically-nearest
PoP (the classic hot-potato baseline). When ``compute_routing_plane``
is invoked with ``demand_per_pair``, the controller additionally
walks each cell's geographic cascade and picks the first PoP whose
aggregate ingress capacity can still absorb the (cell, dst) demand;
if every cascade PoP would overflow, it spills to the least-loaded
cascade PoP. This mirrors the data plane's hot-potato behaviour but
at PoP granularity (not per-sat-feeder), giving an apples-to-apples
control-layer comparison against PG / LP / MILP without pretending
baseline is capacity-blind.

The satellite path is represented by the refresh-time
:class:`SatPathTable` (Dijkstra output) and per-PoP egress candidates
are precomputed into a :class:`PopEgressTable`; both artifacts travel
in the emitted :class:`RoutingPlane` so the data plane never
recomputes routing.
"""

from __future__ import annotations

import time
from types import MappingProxyType
from collections.abc import Mapping

from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_demand_items,
    build_pop_egress_table,
    build_sat_path_table,
    compute_pop_capacity,
    walk_cascade_feasible,
)
from vantage.domain import (
    CellGrid,
    CellToPopTable,
    NetworkSnapshot,
    RoutingPlane,
)


class NearestPoPController:
    """Baseline: route to the closest PoP, spilling to next cascade PoP on overflow."""

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
        demand_per_pair: dict[tuple[str, str], float] | None = None,
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

        # When demand is provided, refine the pure-geographic cascade
        # into a capacity-feasible assignment: first-fit cascade walk
        # with least-loaded-ratio overflow. Only (cell, dst) pairs
        # whose feasible PoP differs from the geographic head need a
        # per-dest override; the rest inherit the geographic cascade
        # from ``mapping`` untouched.
        if demand_per_pair:
            pop_cap = compute_pop_capacity(snapshot)
            items = build_demand_items(demand_per_pair, cell_grid)
            assignments = walk_cascade_feasible(
                cell_to_pop.mapping, items, pop_cap,
            )
            overrides: dict[tuple[int, str], tuple[str, ...]] = {}
            for (cell_id, dst), chosen_pop in assignments.items():
                base_ranked = cell_to_pop.mapping.get(cell_id, ())
                if not base_ranked or base_ranked[0] == chosen_pop:
                    continue
                tail = tuple(p for p in base_ranked if p != chosen_pop)
                overrides[(cell_id, dst)] = (chosen_pop,) + tail
            if overrides:
                cell_to_pop = CellToPopTable(
                    mapping=cell_to_pop.mapping,
                    version=version,
                    built_at=snapshot.time_s,
                    per_dest=MappingProxyType(overrides),
                )
        t2 = perf()

        sat_paths = build_sat_path_table(snapshot, version=version)
        t3 = perf()
        pop_egress = build_pop_egress_table(snapshot, version=version)
        t4 = perf()

        self._last_timing = MappingProxyType({
            "cell_to_pop_ms": (t1 - t0) * 1000.0,
            "capacity_walk_ms": (t2 - t1) * 1000.0,
            "sat_paths_ms": (t3 - t2) * 1000.0,
            "pop_egress_ms": (t4 - t3) * 1000.0,
        })

        return RoutingPlane(
            cell_to_pop=cell_to_pop,
            sat_paths=sat_paths,
            pop_egress=pop_egress,
            version=version,
            built_at=snapshot.time_s,
        )
