"""Shared E2E planning context for capacity-aware control policies."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from vantage.control.costing import build_ground_cost_lookup
from vantage.control.knowledge import GroundKnowledge
from vantage.control.plane import CellToPopTable
from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_demand_items,
    compute_cell_sat_cost,
    compute_pop_capacity,
    rank_pops_by_e2e,
)
from vantage.model import CellGrid, NetworkSnapshot, PoP

GroundCostFn = Callable[[str, str], float | None]
RankedDemandItem = tuple[int, str, float, list[tuple[str, float]]]

__all__ = [
    "E2EPlanningContext",
    "E2EPlanningTiming",
    "GroundCostFn",
    "RankedDemandItem",
    "build_e2e_planning_context",
    "build_policy_ground_cost",
    "build_ranked_demand_items",
    "resolve_policy_dest_names",
]


@dataclass(frozen=True, slots=True)
class E2EPlanningTiming:
    """Monotonic timestamps captured while building a planning context."""

    start: float
    baseline_done: float
    cell_sat_cost_done: float
    rankings_done: float
    pop_cap_done: float
    items_done: float


@dataclass(frozen=True, slots=True)
class E2EPlanningContext:
    """Common inputs used by Greedy, LP-rounding, and MILP controllers."""

    pops: tuple[PoP, ...]
    dest_names: tuple[str, ...]
    ground_cost: GroundCostFn
    baseline: CellToPopTable
    cell_sat_cost: dict[tuple[int, str], float]
    rankings: dict[tuple[int, str], list[tuple[str, float]]]
    pop_cap: dict[str, float]
    items: list[RankedDemandItem]
    timing: E2EPlanningTiming


def resolve_policy_dest_names(
    *,
    controller_name: str,
    explicit_dest_names: tuple[str, ...],
    ground_knowledge: GroundKnowledge,
    warned_no_dests: bool,
    logger: logging.Logger,
) -> tuple[tuple[str, ...], bool]:
    """Resolve a policy's planning destinations and emit one empty-cache warning."""
    if explicit_dest_names:
        return explicit_dest_names, warned_no_dests

    derived = tuple(sorted({dest for _, dest in ground_knowledge.all_entries()}))
    if not derived and not warned_no_dests:
        logger.warning(
            "%s.resolve_dest_names: no explicit dest_names and "
            "ground_knowledge has no entries; degrading to nearest-PoP "
            "baseline this epoch.",
            controller_name,
        )
        warned_no_dests = True
    return derived, warned_no_dests


def build_policy_ground_cost(
    ground_knowledge: GroundKnowledge,
    *,
    current_epoch: int,
    lambda_dev: float,
    stale_per_epoch_ms: float,
    pops: Iterable[PoP] | None = None,
    dest_names: Iterable[str] | None = None,
) -> GroundCostFn:
    """Build the scoring surface shared by E2E-aware policy planners."""
    return build_ground_cost_lookup(
        ground_knowledge,
        current_epoch=current_epoch,
        lambda_dev=lambda_dev,
        stale_per_epoch_ms=stale_per_epoch_ms,
        pops=pops,
        dest_names=dest_names,
    )


def build_ranked_demand_items(
    rankings: Mapping[tuple[int, str], list[tuple[str, float]]],
    cell_grid: CellGrid,
    demand_per_pair: Mapping[tuple[str, str], float],
) -> list[RankedDemandItem]:
    """Attach ranked PoP cascades to aggregated ``(cell, dest)`` demand."""
    demand_by_cell_dest = {
        (cell_id, dest): demand
        for cell_id, dest, demand in build_demand_items(demand_per_pair, cell_grid)
    }
    items: list[RankedDemandItem] = []
    for key, ranked in rankings.items():
        if not ranked:
            continue
        demand = demand_by_cell_dest.get(key, 0.0)
        if demand <= 0.0:
            continue
        cell_id, dest = key
        items.append((cell_id, dest, demand, ranked))
    return items


def build_e2e_planning_context(
    *,
    snapshot: NetworkSnapshot,
    cell_grid: CellGrid,
    ground_knowledge: GroundKnowledge,
    dest_names: tuple[str, ...],
    demand_per_pair: Mapping[tuple[str, str], float],
    version: int,
    score_lambda_dev: float,
    score_stale_per_epoch_ms: float,
    include_items: bool,
) -> E2EPlanningContext:
    """Build the common E2E ranking and capacity inputs for a policy refresh."""
    perf = time.perf_counter
    pops = snapshot.infra.pops

    t0 = perf()
    ground_cost = build_policy_ground_cost(
        ground_knowledge,
        current_epoch=int(version),
        lambda_dev=score_lambda_dev,
        stale_per_epoch_ms=score_stale_per_epoch_ms,
        pops=pops,
        dest_names=dest_names,
    )

    baseline = build_cell_to_pop_nearest(
        cell_grid=cell_grid,
        pops=pops,
        built_at=snapshot.time_s,
        version=version,
    )
    t_baseline = perf()

    cell_sat_cost = compute_cell_sat_cost(snapshot, cell_grid)
    t_cell_sat = perf()

    rankings = rank_pops_by_e2e(
        cell_grid=cell_grid,
        pops=pops,
        baseline=baseline,
        cell_sat_cost=cell_sat_cost,
        ground_cost_fn=ground_cost,
        dest_names=dest_names,
    )
    t_rankings = perf()

    pop_cap = compute_pop_capacity(snapshot)
    t_pop_cap = perf()

    items = (
        build_ranked_demand_items(rankings, cell_grid, demand_per_pair)
        if include_items
        else []
    )
    t_items = perf()

    return E2EPlanningContext(
        pops=pops,
        dest_names=dest_names,
        ground_cost=ground_cost,
        baseline=baseline,
        cell_sat_cost=cell_sat_cost,
        rankings=rankings,
        pop_cap=pop_cap,
        items=items,
        timing=E2EPlanningTiming(
            start=t0,
            baseline_done=t_baseline,
            cell_sat_cost_done=t_cell_sat,
            rankings_done=t_rankings,
            pop_cap_done=t_pop_cap,
            items_done=t_items,
        ),
    )
