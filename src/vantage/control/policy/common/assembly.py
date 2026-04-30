"""Shared RoutingPlane assembly helpers for policy controllers."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from vantage.control.plane import CellToPopTable, RoutingPlane
from vantage.control.policy.common.fib_builder import (
    build_pop_egress_table,
    build_sat_path_table,
)
from vantage.model import NetworkSnapshot

__all__ = [
    "RoutingPlaneAssembly",
    "assemble_assignment_routing_plane",
    "build_assignment_cell_to_pop",
]


@dataclass(frozen=True, slots=True)
class RoutingPlaneAssembly:
    """Routing plane plus per-step assembly timings in milliseconds."""

    plane: RoutingPlane
    cell_to_pop: CellToPopTable
    timing_ms: Mapping[str, float]


def build_assignment_cell_to_pop(
    *,
    baseline: CellToPopTable,
    rankings: Mapping[tuple[int, str], list[tuple[str, float]]],
    assignments: Mapping[tuple[int, str], str],
    version: int,
    built_at: float,
) -> CellToPopTable:
    """Overlay chosen per-destination PoPs on a baseline cell cascade."""
    per_dest_overrides: dict[tuple[int, str], tuple[str, ...]] = {}
    baseline_mapping = baseline.mapping
    for (cell_id, dest), ranked in rankings.items():
        if not ranked:
            continue
        chosen_pop = assignments.get((cell_id, dest))
        if chosen_pop is None:
            base_ranked = baseline_mapping.get(cell_id)
            if not base_ranked:
                continue
            chosen_pop = base_ranked[0]
        tail = tuple(pop for pop, _cost in ranked if pop != chosen_pop)
        ranked_tuple = (chosen_pop,) + tail
        if ranked_tuple != baseline_mapping.get(cell_id, ()):
            per_dest_overrides[(cell_id, dest)] = ranked_tuple

    return CellToPopTable(
        mapping=baseline_mapping,
        version=version,
        built_at=built_at,
        per_dest=MappingProxyType(per_dest_overrides),
    )


def assemble_assignment_routing_plane(
    *,
    snapshot: NetworkSnapshot,
    baseline: CellToPopTable,
    rankings: Mapping[tuple[int, str], list[tuple[str, float]]],
    assignments: Mapping[tuple[int, str], str],
    version: int,
) -> RoutingPlaneAssembly:
    """Assemble a full routing plane from a policy assignment."""
    perf = time.perf_counter
    t0 = perf()
    cell_to_pop = build_assignment_cell_to_pop(
        baseline=baseline,
        rankings=rankings,
        assignments=assignments,
        version=version,
        built_at=snapshot.time_s,
    )
    t_cell_to_pop = perf()
    sat_paths = build_sat_path_table(snapshot, version=version)
    t_sat_paths = perf()
    pop_egress = build_pop_egress_table(snapshot, version=version)
    t_pop_egress = perf()

    plane = RoutingPlane(
        cell_to_pop=cell_to_pop,
        sat_paths=sat_paths,
        pop_egress=pop_egress,
        version=version,
        built_at=snapshot.time_s,
    )
    timing = MappingProxyType({
        "cell_to_pop_ms": (t_cell_to_pop - t0) * 1000.0,
        "sat_paths_ms": (t_sat_paths - t_cell_to_pop) * 1000.0,
        "pop_egress_ms": (t_pop_egress - t_sat_paths) * 1000.0,
    })
    return RoutingPlaneAssembly(
        plane=plane,
        cell_to_pop=cell_to_pop,
        timing_ms=timing,
    )
