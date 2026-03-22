"""Data plane: realize RoutingIntent against network truth.

Layer boundary: **forward** is the execution layer. It does NOT search
or decide — it only computes actual delays along pre-resolved paths.
Controller outputs fully resolved paths (pop, gs, user_sat, egress_sat).
Forward computes the actual delays. All delays in ms.

Ground delay resolution uses context.ground_knowledge (unified service):
L1 cache hit → use cached value; cache miss → L2/L3 estimator fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vantage.domain import (
    EpochResult,
    FlowOutcome,
    NetworkSnapshot,
    RoutingIntent,
    TrafficDemand,
)

if TYPE_CHECKING:
    from vantage.engine.context import RunContext


def realize(
    intent: RoutingIntent,
    snapshot: NetworkSnapshot,
    demand: TrafficDemand,
    context: RunContext,
) -> EpochResult:
    """Realize a RoutingIntent: compute E2E delays along resolved paths.

    satellite_rtt = uplink + ISL + downlink + backhaul (calibrated), ms.
    ground_rtt = PoP→destination (from ground_knowledge), ms.
    """
    sat = snapshot.satellite
    gk = context.ground_knowledge
    outcomes: list[FlowOutcome] = []
    total_demand = 0.0
    routed_demand = 0.0
    calibration = context.world.calibration

    for flow_key, alloc in intent.allocations.items():
        flow_demand = demand.flows.get(flow_key, 0.0)
        total_demand += flow_demand

        src_ep = context.endpoints.get(flow_key.src)
        if src_ep is None:
            continue

        gs = snapshot.infra.gs_by_id(alloc.gs_id)
        if gs is None:
            continue

        # Satellite propagation RTT (uplink + ISL + downlink) × 2
        raw_propagation_rtt = sat.compute_satellite_rtt(
            alloc.user_sat, alloc.egress_sat,
            src_ep.lat_deg, src_ep.lon_deg,
            gs.lat_deg, gs.lon_deg,
        )

        # Calibrate satellite propagation
        if calibration is not None:
            propagation_rtt = calibration.calibrate(
                flow_key.src, raw_propagation_rtt
            )
        else:
            propagation_rtt = raw_propagation_rtt

        # Backhaul RTT (GS↔PoP)
        backhaul_rtt = snapshot.infra.get_backhaul_delay(
            alloc.gs_id, alloc.pop_code
        ) * 2

        # satellite_rtt = propagation + backhaul (everything terminal→PoP)
        satellite_rtt = propagation_rtt + backhaul_rtt

        # Ground RTT (PoP→destination) via unified knowledge service
        pop = snapshot.infra.pop_by_code(alloc.pop_code)
        dst_ep = context.endpoints.get(flow_key.dst)
        if pop is not None and dst_ep is not None:
            ground_rtt = gk.get_or_estimate(
                alloc.pop_code, flow_key.dst,
                pop.lat_deg, pop.lon_deg,
                dst_ep.lat_deg, dst_ep.lon_deg,
            )
        else:
            ground_rtt = gk.get(alloc.pop_code, flow_key.dst) or 0.0

        total_rtt = satellite_rtt + ground_rtt

        outcomes.append(FlowOutcome(
            flow_key=flow_key,
            pop_code=alloc.pop_code,
            gs_id=alloc.gs_id,
            user_sat=alloc.user_sat,
            egress_sat=alloc.egress_sat,
            satellite_rtt=satellite_rtt,
            ground_rtt=ground_rtt,
            total_rtt=total_rtt,
            demand_gbps=flow_demand,
        ))
        routed_demand += flow_demand

    # Count unrouted flows
    for flow_key, flow_demand in demand.flows.items():
        if flow_key not in intent.allocations:
            total_demand += flow_demand

    return EpochResult(
        epoch=intent.epoch,
        flow_outcomes=tuple(outcomes),
        total_demand_gbps=total_demand,
        routed_demand_gbps=routed_demand,
        unrouted_demand_gbps=total_demand - routed_demand,
    )
