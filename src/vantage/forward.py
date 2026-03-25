"""Data plane: terminal-side PoP selection + delay computation.

Two separate logics:
1. DECISION: terminal selects PoP from cost tables.
   - ground_cost present → use it for joint optimization
   - ground_cost missing → fallback to nearest PoP (sat_cost only, like BGP)
2. MEASUREMENT: compute actual E2E delay along resolved path.
   - ground_rtt always computed from ground_truth model (physical reality)
   - independent of whether controller had data for this pair

All delays in ms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vantage.common import haversine_km
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

# Sentinel: ground_cost not available → fallback to sat_cost only
_NO_GROUND_DATA = -1.0


def realize(
    tables: CostTables,
    snapshot: NetworkSnapshot,
    demand: TrafficDemand,
    context: RunContext,
) -> EpochResult:
    """Terminal-side PoP selection + actual E2E delay computation.

    Decision: argmin(sat_cost + ground_cost) where ground_cost is available.
              For PoPs without ground_cost data, they don't participate in
              joint optimization — terminal falls back to lowest sat_cost.

    Measurement: actual ground_rtt computed from ground_truth model
                 (HaversineDelay), representing physical reality.
    """
    sat = snapshot.satellite
    calibration = context.world.calibration
    # Ground truth model for actual delay computation (NOT for decisions)
    ground_truth = context.ground_knowledge.estimator
    outcomes: list[FlowOutcome] = []
    total_demand = 0.0
    routed_demand = 0.0

    for flow_key, flow_demand in demand.flows.items():
        total_demand += flow_demand

        src_ep = context.endpoints.get(flow_key.src)
        if src_ep is None:
            continue

        uplink = find_ingress_satellite(src_ep, sat.positions)
        if uplink is None:
            continue

        ingress = uplink.sat_id

        # ── DECISION: select best PoP ──────────────────────
        # Two-tier selection:
        #   1st: try joint optimization (PoPs with ground_cost data)
        #   2nd: fallback to sat_cost only (nearest PoP, like BGP default)
        best_pop: str | None = None
        best_cost = float("inf")
        fallback_pop: str | None = None
        fallback_cost = float("inf")

        for pop in snapshot.infra.pops:
            sc = tables.sat_cost.get((ingress, pop.code))
            if sc is None:
                continue

            # Track best fallback (sat_cost only, for BGP default)
            if sc < fallback_cost:
                fallback_cost = sc
                fallback_pop = pop.code

            # Joint optimization: only if ground_cost data exists
            gc = tables.ground_cost.get((pop.code, flow_key.dst))
            if gc is not None and gc >= 0:
                total = sc + gc
                if total < best_cost:
                    best_cost = total
                    best_pop = pop.code

        # If no PoP had ground data → fallback to nearest (BGP)
        if best_pop is None:
            best_pop = fallback_pop

        if best_pop is None:
            continue

        # ── RESOLVE: best (GS, egress_sat) for chosen PoP ──
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

        # ── MEASUREMENT: actual ground RTT (physical truth) ──
        # This is what actually happens — independent of controller knowledge.
        # Uses ground_truth model (e.g., HaversineDelay) to simulate real delay.
        pop_obj = snapshot.infra.pop_by_code(best_pop)
        dst_ep = context.endpoints.get(flow_key.dst)
        if pop_obj is not None and dst_ep is not None and ground_truth is not None:
            ground_rtt = ground_truth.estimate(
                pop_obj.lat_deg, pop_obj.lon_deg,
                dst_ep.lat_deg, dst_ep.lon_deg,
            ) * 2  # one-way → RTT
        else:
            ground_rtt = 0.0

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
