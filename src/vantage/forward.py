"""Data plane: terminal-side PoP selection + delay computation.

Two separate logics:
1. DECISION: terminal selects PoP from cost tables.
   - ground_cost present → use it for joint optimization
   - ground_cost missing → fallback to nearest PoP (sat_cost only, like BGP)
2. MEASUREMENT: compute actual E2E delay along resolved path.
   - ground_rtt always computed from ground_truth measurement table
     (physical reality); unknown pairs drop the flow.

All delays in ms.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from vantage.common import DEFAULT_MIN_ELEVATION_DEG
from vantage.common.geo import access_delay
from vantage.common.time import resolve_local_time
from vantage.control.policy.common.utils import find_ingress_satellite
from vantage.world.satellite.visibility import SphericalAccessModel
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
    *,
    epoch_interval_s: float = 3600.0,
    simulation_start_utc: datetime | None = None,
) -> EpochResult:
    """Terminal-side PoP selection + actual E2E delay computation.

    Decision: argmin(sat_cost + ground_cost) where ground_cost is available.
              For PoPs without ground_cost data, they don't participate in
              joint optimization — terminal falls back to lowest sat_cost.

    Measurement: actual ground_rtt pulled from the ``MeasuredGroundDelay``
                 table behind ``ground_knowledge.estimator``. Unknown
                 pairs raise KeyError and drop the flow — no fallback.
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

        # ── RESOLVE: best satellite path for chosen PoP ──
        # Priority 1: bent-pipe — find a satellite visible from BOTH
        #   the terminal and a GS of this PoP (ISL = 0).
        # Priority 2: ISL relay — fixed ingress + different egress via ISL.
        best_gs: str | None = None
        best_egress: int = -1
        best_ingress: int = ingress
        best_sat_rtt = float("inf")

        _access = SphericalAccessModel()

        for gs_id, backhaul in snapshot.infra.pop_gs_edges(best_pop):
            gs = snapshot.infra.gs_by_id(gs_id)
            if gs is None:
                continue
            gs_links = sat.gateway_attachments.attachments.get(gs_id)
            if not gs_links:
                continue
            backhaul_rtt = backhaul * 2

            for link in gs_links:
                egress_id = link.sat_id

                # Check bent-pipe: is this GS satellite also visible from terminal?
                elev = _access.compute_access_pair(
                    src_ep.lat_deg, src_ep.lon_deg, 0.0,
                    float(sat.positions[egress_id, 0]),
                    float(sat.positions[egress_id, 1]),
                    float(sat.positions[egress_id, 2]),
                ).elevation_deg

                if elev >= DEFAULT_MIN_ELEVATION_DEG:
                    # Bent-pipe: terminal and GS share this satellite
                    up = access_delay(
                        src_ep.lat_deg, src_ep.lon_deg,
                        float(sat.positions[egress_id, 0]),
                        float(sat.positions[egress_id, 1]),
                        float(sat.positions[egress_id, 2]),
                    )
                    down = access_delay(
                        gs.lat_deg, gs.lon_deg,
                        float(sat.positions[egress_id, 0]),
                        float(sat.positions[egress_id, 1]),
                        float(sat.positions[egress_id, 2]),
                    )
                    raw_prop = (up + down) * 2  # RTT, no ISL
                else:
                    # ISL relay: use fixed ingress → ISL → egress
                    raw_prop = sat.compute_satellite_rtt(
                        ingress, egress_id,
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
                    best_egress = egress_id
                    best_ingress = egress_id if elev >= DEFAULT_MIN_ELEVATION_DEG else ingress

        if best_gs is None:
            continue

        satellite_rtt = best_sat_rtt

        # ── MEASUREMENT: actual ground RTT (physical truth) ──
        # Measured from ``ground_knowledge.estimator`` (a strict
        # ``MeasuredGroundDelay`` table). Unknown (pop, dest) pairs
        # raise KeyError and drop the flow here — there is no
        # geographic fallback in this codebase anymore.
        pop_obj = snapshot.infra.pop_by_code(best_pop)

        if ground_truth is not None and pop_obj is not None:
            try:
                ground_rtt = ground_truth.estimate(best_pop, flow_key.dst) * 2
            except KeyError:
                # No measured (pop, dest) pair — flow is unrouted.
                continue
        elif context.service_ground_delay is not None and pop_obj is not None:
            # Service-class destination: use profiled model
            local_hour, day_type = resolve_local_time(
                demand.epoch, epoch_interval_s, simulation_start_utc,
                getattr(context.service_ground_delay, "pop_timezones", {}),
                best_pop,
            )
            ground_rtt = context.service_ground_delay.estimate_service(
                best_pop, flow_key.dst, local_hour, day_type,
            )
        else:
            continue

        total_rtt = satellite_rtt + ground_rtt

        outcomes.append(FlowOutcome(
            flow_key=flow_key,
            pop_code=best_pop,
            gs_id=best_gs,
            user_sat=best_ingress,
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
