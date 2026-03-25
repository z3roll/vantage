"""Data plane: terminal-side PoP selection + delay computation.

Controller provides CostTables (precomputed sat_cost + ground_cost).
For each flow, the terminal selects the best PoP via:
    argmin over pop: sat_cost[ingress_sat, pop] + ground_cost[pop, dest]

Then actual E2E delays are computed along the resolved path.
All delays in ms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vantage.control.policy.common.utils import find_ingress_satellite
from vantage.domain import (
    CostTables,
    EpochResult,
    FlowOutcome,
    NetworkSnapshot,
    TrafficDemand,
)

if TYPE_CHECKING:
    from vantage.engine.context import RunContext


def realize(
    tables: CostTables,
    snapshot: NetworkSnapshot,
    demand: TrafficDemand,
    context: RunContext,
) -> EpochResult:
    """Realize CostTables: terminal-side PoP selection + E2E delay computation.

    For each flow:
    1. Find ingress satellite (best visible from terminal)
    2. Select best PoP: argmin(sat_cost + ground_cost)
    3. Resolve best (GS, egress_sat) for that PoP
    4. Compute actual segment delays
    """
    sat = snapshot.satellite
    gk = context.ground_knowledge
    calibration = context.world.calibration
    outcomes: list[FlowOutcome] = []
    total_demand = 0.0
    routed_demand = 0.0

    for flow_key, flow_demand in demand.flows.items():
        total_demand += flow_demand

        src_ep = context.endpoints.get(flow_key.src)
        if src_ep is None:
            continue

        # Terminal finds its ingress satellite
        uplink = find_ingress_satellite(src_ep, sat.positions)
        if uplink is None:
            continue

        ingress = uplink.sat_id

        # Terminal-side PoP selection: argmin(sat_cost + ground_cost)
        best_pop: str | None = None
        best_total_cost = float("inf")
        for pop in snapshot.infra.pops:
            sc = tables.sat_cost.get((ingress, pop.code))
            gc = tables.ground_cost.get((pop.code, flow_key.dst), 0.0)
            if sc is None:
                continue
            total = sc + gc
            if total < best_total_cost:
                best_total_cost = total
                best_pop = pop.code

        if best_pop is None:
            continue

        # Resolve best (GS, egress_sat) for the chosen PoP
        best_gs: str | None = None
        best_egress: int = -1
        best_sat_rtt = float("inf")

        for gs_id, backhaul in snapshot.infra.pop_gs_edges(best_pop):
            gs = snapshot.infra.gs_by_id(gs_id)
            if gs is None:
                continue
            gs_links = sat.gateway_attachments.attachments.get(gs_id)
            if not gs_links:
                continue
            backhaul_rtt = backhaul * 2
            for link in gs_links:
                raw_prop = sat.compute_satellite_rtt(
                    ingress, link.sat_id,
                    src_ep.lat_deg, src_ep.lon_deg,
                    gs.lat_deg, gs.lon_deg,
                )
                if calibration is not None:
                    prop = calibration.calibrate(flow_key.src, raw_prop)
                else:
                    prop = raw_prop
                total_sat = prop + backhaul_rtt
                if total_sat < best_sat_rtt:
                    best_sat_rtt = total_sat
                    best_gs = gs_id
                    best_egress = link.sat_id

        if best_gs is None:
            continue

        satellite_rtt = best_sat_rtt

        # Ground RTT from knowledge service
        pop_obj = snapshot.infra.pop_by_code(best_pop)
        dst_ep = context.endpoints.get(flow_key.dst)
        if pop_obj is not None and dst_ep is not None:
            ground_rtt = gk.get_or_estimate(
                best_pop, flow_key.dst,
                pop_obj.lat_deg, pop_obj.lon_deg,
                dst_ep.lat_deg, dst_ep.lon_deg,
            )
        else:
            ground_rtt = gk.get(best_pop, flow_key.dst) or 0.0

        total_rtt = satellite_rtt + ground_rtt

        outcomes.append(FlowOutcome(
            flow_key=flow_key,
            pop_code=best_pop,
            gs_id=best_gs,
            user_sat=ingress,
            egress_sat=best_egress,
            satellite_rtt=satellite_rtt,
            ground_rtt=ground_rtt,
            total_rtt=total_rtt,
            demand_gbps=flow_demand,
        ))
        routed_demand += flow_demand

    return EpochResult(
        epoch=tables.epoch,
        flow_outcomes=tuple(outcomes),
        total_demand_gbps=total_demand,
        routed_demand_gbps=routed_demand,
        unrouted_demand_gbps=total_demand - routed_demand,
    )
