"""Control-plan evaluation helpers shared by simulation and tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from vantage.common.stats import weighted_mean, weighted_percentile
from vantage.control.costing import build_ground_cost_lookup
from vantage.control.knowledge import GroundKnowledge
from vantage.control.plane import RoutingPlane
from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    compute_cell_sat_cost,
    compute_pop_capacity,
    rank_pops_by_e2e,
)
from vantage.control.policy.common.planning import (
    RankedDemandItem,
    build_ranked_demand_items,
)
from vantage.model import CellGrid, NetworkSnapshot

__all__ = [
    "ControlPlanEvaluation",
    "RankedDemandItem",
    "assignment_from_routing_plane",
    "build_ranked_demand_items",
    "compute_assignment_objective",
    "evaluate_control_plans",
    "summarize_plan_latency",
]


@dataclass(frozen=True, slots=True)
class ControlPlanEvaluation:
    """Dashboard-ready control-plan metrics for one refresh epoch."""

    plan_costs: Mapping[str, float | None] = field(default_factory=dict)
    plan_stats: Mapping[str, Mapping[str, float] | None] = field(default_factory=dict)
    plan_lb: float | None = None
    plan_lb_mean: float | None = None

    @classmethod
    def empty(cls) -> ControlPlanEvaluation:
        return cls()

    def dashboard_fields(self, labels: Iterable[str] = ("bl", "greedy", "lp", "mip")) -> dict[str, float | None]:
        out: dict[str, float | None] = {
            "plan_lb": self.plan_lb,
            "plan_lb_mean": self.plan_lb_mean,
        }
        for label in labels:
            out[f"{label}_plan_cost"] = self.plan_costs.get(label)
            stats = self.plan_stats.get(label) or {}
            for metric in ("mean", "sat", "gnd", "p95", "p99"):
                out[f"{label}_plan_{metric}"] = stats.get(metric)
        return out


def assignment_from_routing_plane(
    plane: RoutingPlane,
    ranking_keys: Iterable[tuple[int, str]],
) -> dict[tuple[int, str], str]:
    """Extract primary ``(cell, dest) -> pop`` choices from a routing plane."""
    per_dest = plane.cell_to_pop.per_dest
    mapping = plane.cell_to_pop.mapping
    out: dict[tuple[int, str], str] = {}
    for key in ranking_keys:
        override = per_dest.get(key)
        if override:
            out[key] = override[0]
            continue
        base = mapping.get(key[0])
        if base:
            out[key] = base[0]
    return out


def compute_assignment_objective(
    assignments: Mapping[tuple[int, str], str],
    items: Iterable[RankedDemandItem],
    pop_cap: Mapping[str, float],
    *,
    overflow_penalty: float,
) -> float:
    """Compute ``sum(demand * cost) + overflow_penalty * overflow``."""
    idx: dict[tuple[int, str], tuple[float, list[tuple[str, float]]]] = {
        (cell_id, dest): (demand, ranked)
        for cell_id, dest, demand, ranked in items
    }
    if not idx:
        return 0.0

    cost = 0.0
    pop_load: dict[str, float] = {}
    for key, pop in assignments.items():
        item = idx.get(key)
        if item is None:
            continue
        demand, ranked = item
        for candidate_pop, candidate_cost in ranked:
            if candidate_pop == pop:
                cost += demand * candidate_cost
                break
        pop_load[pop] = pop_load.get(pop, 0.0) + demand

    for pop, load in pop_load.items():
        cap = pop_cap.get(pop, 0.0)
        if cap > 0.0 and load > cap:
            cost += overflow_penalty * (load - cap)
    return cost


def summarize_plan_latency(
    assignments: Mapping[tuple[int, str], str],
    items: Iterable[RankedDemandItem],
    cell_sat_cost: Mapping[tuple[int, str], float],
    ground_cost: Callable[[str, str], float | None],
) -> dict[str, float]:
    """Demand-weighted predicted RTT stats for one control assignment."""
    rtts: list[float] = []
    sats: list[float] = []
    gnds: list[float] = []
    demands: list[float] = []

    for cell_id, dest, demand, _ranked in items:
        pop = assignments.get((cell_id, dest))
        if pop is None:
            continue
        sat = cell_sat_cost.get((cell_id, pop))
        if sat is None:
            continue
        gnd = ground_cost(pop, dest) or 0.0
        rtts.append(sat + gnd)
        sats.append(sat)
        gnds.append(gnd)
        demands.append(demand)

    if not rtts:
        return {"mean": 0.0, "sat": 0.0, "gnd": 0.0, "p95": 0.0, "p99": 0.0}

    pairs = list(zip(rtts, demands, strict=True))
    return {
        "mean": weighted_mean(rtts, demands),
        "sat": weighted_mean(sats, demands),
        "gnd": weighted_mean(gnds, demands),
        "p95": weighted_percentile(pairs, 95),
        "p99": weighted_percentile(pairs, 99),
    }


def evaluate_control_plans(
    *,
    snapshot: NetworkSnapshot,
    cell_grid: CellGrid,
    demand_per_pair: Mapping[tuple[str, str], float],
    routing_planes: Mapping[str, RoutingPlane],
    reference_ground_knowledge: GroundKnowledge,
    dest_names: Iterable[str],
    current_epoch: int,
    lp_lower_bound: float | None,
    lambda_dev: float = 1.0,
    stale_per_epoch_ms: float = 0.05,
    overflow_penalty: float = 1.0e4,
) -> ControlPlanEvaluation:
    """Evaluate multiple routing planes on a shared control objective."""
    ground_cost = build_ground_cost_lookup(
        reference_ground_knowledge,
        current_epoch=current_epoch,
        lambda_dev=lambda_dev,
        stale_per_epoch_ms=stale_per_epoch_ms,
        pops=snapshot.infra.pops,
        dest_names=dest_names,
    )
    baseline = build_cell_to_pop_nearest(
        cell_grid=cell_grid,
        pops=snapshot.infra.pops,
        built_at=snapshot.time_s,
        version=current_epoch,
    )
    cell_sat_cost = compute_cell_sat_cost(snapshot, cell_grid)
    rankings = rank_pops_by_e2e(
        cell_grid=cell_grid,
        pops=snapshot.infra.pops,
        baseline=baseline,
        cell_sat_cost=cell_sat_cost,
        ground_cost_fn=ground_cost,
        dest_names=dest_names,
    )
    pop_cap = compute_pop_capacity(snapshot)
    items = build_ranked_demand_items(rankings, cell_grid, demand_per_pair)

    plan_costs: dict[str, float | None] = {}
    plan_stats: dict[str, dict[str, float] | None] = {}
    for label, plane in routing_planes.items():
        assignments = assignment_from_routing_plane(plane, rankings)
        plan_costs[label] = round(
            compute_assignment_objective(
                assignments,
                items,
                pop_cap,
                overflow_penalty=overflow_penalty,
            ),
            2,
        )
        stats = summarize_plan_latency(
            assignments,
            items,
            cell_sat_cost,
            ground_cost,
        )
        plan_stats[label] = {k: round(v, 3) for k, v in stats.items()}

    plan_lb = round(lp_lower_bound, 2) if lp_lower_bound is not None else None
    if plan_lb is not None and items:
        total_demand = sum(demand for _, _, demand, _ in items)
        plan_lb_mean = round(plan_lb / total_demand, 3) if total_demand > 0 else None
    else:
        plan_lb_mean = None

    return ControlPlanEvaluation(
        plan_costs=plan_costs,
        plan_stats=plan_stats,
        plan_lb=plan_lb,
        plan_lb_mean=plan_lb_mean,
    )
