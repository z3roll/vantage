"""Ground-delay progressive spillover baseline.

For each destination service, rank PoPs by ground-segment RTT only.
Then process atomic ``(cell, destination)`` demand items in descending
demand order and place each item at the first PoP in that destination's
ground-latency ranking whose aggregate PoP capacity can absorb the
whole item.

This is deliberately simpler than :class:`GreedyController`: it does
not compute E2E rankings, improvement over nearest-PoP, or an LP/MILP
objective. It models a plausible service-centric GSLB spillover rule:
"send the largest demand blocks to the lowest ground-latency PoP for
that service until the PoP is full, then spill to the next one."
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from vantage.control.knowledge import GroundKnowledge
from vantage.control.plane import RoutingPlane
from vantage.control.policy.common.assembly import assemble_assignment_routing_plane
from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_demand_items,
    compute_pop_capacity,
)
from vantage.control.policy.common.planning import (
    build_policy_ground_cost,
    resolve_policy_dest_names,
)
from vantage.model import CellGrid, NetworkSnapshot, PoP

_log = logging.getLogger(__name__)


class ProgressiveSpilloverController:
    """Destination-ground-latency first-fit spillover baseline."""

    def __init__(
        self,
        ground_knowledge: GroundKnowledge | None = None,
        dest_names: tuple[str, ...] = (),
    ) -> None:
        self._gk = ground_knowledge or GroundKnowledge()
        self._dest_names = dest_names
        self._warned_no_dests = False
        self._last_timing: Mapping[str, float] = MappingProxyType({})

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    @property
    def last_timing(self) -> Mapping[str, float]:
        """Step timings (ms) from the most recent plan build."""
        return self._last_timing

    def resolve_dest_names(self) -> tuple[str, ...]:
        dest_names, self._warned_no_dests = resolve_policy_dest_names(
            controller_name="ProgressiveSpilloverController",
            explicit_dest_names=self._dest_names,
            ground_knowledge=self._gk,
            warned_no_dests=self._warned_no_dests,
            logger=_log,
        )
        return dest_names

    def _make_ground_cost(
        self,
        *,
        current_epoch: int,
        pops: Iterable[PoP] | None = None,
        dest_names: Iterable[str] | None = None,
    ):
        """Compatibility shim mirroring other policy controllers."""
        return build_policy_ground_cost(
            self._gk,
            current_epoch=current_epoch,
            lambda_dev=0.0,
            stale_per_epoch_ms=0.0,
            pops=pops,
            dest_names=dest_names,
        )

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
        dest_names = self.resolve_dest_names()
        pops = snapshot.infra.pops
        ground_cost = self._make_ground_cost(
            current_epoch=version,
            pops=pops,
            dest_names=dest_names,
        )
        baseline = build_cell_to_pop_nearest(
            cell_grid=cell_grid,
            pops=pops,
            built_at=snapshot.time_s,
            version=version,
        )
        demand_items = build_demand_items(demand_per_pair or {}, cell_grid)
        t_baseline = perf()

        ranked_by_dest = _rank_pops_by_ground_delay(
            pops=pops,
            dest_names=dest_names,
            ground_cost=ground_cost,
        )
        rankings = _expand_rankings_for_items(
            items=demand_items,
            ranked_by_dest=ranked_by_dest,
        )
        t_rankings = perf()

        pop_cap = compute_pop_capacity(snapshot)
        t_pop_cap = perf()

        assignments = _progressive_assign(
            items=demand_items,
            ranked_by_dest=ranked_by_dest,
            pop_cap=pop_cap,
        )
        t_fill = perf()

        assembly = assemble_assignment_routing_plane(
            snapshot=snapshot,
            baseline=baseline,
            rankings=rankings,
            assignments=assignments,
            version=version,
        )

        self._last_timing = MappingProxyType({
            "baseline_ms": (t_baseline - t0) * 1000.0,
            "ground_rankings_ms": (t_rankings - t_baseline) * 1000.0,
            "pop_cap_ms": (t_pop_cap - t_rankings) * 1000.0,
            "progressive_fill_ms": (
                (t_fill - t_pop_cap) * 1000.0
                + assembly.timing_ms["cell_to_pop_ms"]
            ),
            "sat_paths_ms": assembly.timing_ms["sat_paths_ms"],
            "pop_egress_ms": assembly.timing_ms["pop_egress_ms"],
        })
        return assembly.plane


def _rank_pops_by_ground_delay(
    *,
    pops: tuple[PoP, ...],
    dest_names: tuple[str, ...],
    ground_cost,
) -> dict[str, list[tuple[str, float]]]:
    ranked_by_dest: dict[str, list[tuple[str, float]]] = {}
    for dest in dest_names:
        ranked: list[tuple[str, float]] = []
        for pop in pops:
            cost = ground_cost(pop.code, dest)
            if cost is not None:
                ranked.append((pop.code, float(cost)))
        ranked.sort(key=lambda item: (item[1], item[0]))
        ranked_by_dest[dest] = ranked
    return ranked_by_dest


def _expand_rankings_for_items(
    *,
    items: list[tuple[int, str, float]],
    ranked_by_dest: Mapping[str, list[tuple[str, float]]],
) -> dict[tuple[int, str], list[tuple[str, float]]]:
    rankings: dict[tuple[int, str], list[tuple[str, float]]] = {}
    for cell_id, dest, _demand in items:
        ranked = ranked_by_dest.get(dest)
        if not ranked:
            continue
        rankings[(cell_id, dest)] = ranked
    return rankings


def _progressive_assign(
    *,
    items: list[tuple[int, str, float]],
    ranked_by_dest: Mapping[str, list[tuple[str, float]]],
    pop_cap: Mapping[str, float],
) -> dict[tuple[int, str], str]:
    load: dict[str, float] = {}
    assignments: dict[tuple[int, str], str] = {}
    items_by_priority = sorted(items, key=lambda item: (-item[2], item[1], item[0]))
    for cell_id, dest, demand in items_by_priority:
        for pop_code, _ground_cost in ranked_by_dest.get(dest, ()):
            cap = pop_cap.get(pop_code, 0.0)
            if cap <= 0.0:
                continue
            if load.get(pop_code, 0.0) + demand <= cap:
                assignments[(cell_id, dest)] = pop_code
                load[pop_code] = load.get(pop_code, 0.0) + demand
                break
    return assignments
