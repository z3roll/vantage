"""Data plane for the routing-plane execution path.

This is the *new* forward function introduced alongside the existing
:mod:`vantage.forward`. It consumes a :class:`RoutingPlane` (produced
by a controller's ``compute_routing_plane``) instead of the legacy
``CostTables``, walks each flow's path hop-by-hop through the
per-satellite FIBs, and charges its demand onto a per-epoch
:class:`UsageBook` so capacity utilization and SLA bookkeeping become
first-class outputs.

Scope for this phase:

    * **Decision**: ``cell_to_pop[grid.cell_of(flow.src)]`` picks the
      destination PoP. No PoP re-selection happens inside the data
      plane — the controller already committed a decision when it
      built the plane.
    * **Path walk**: starting from the ingress satellite (user-facing
      nearest sat), repeatedly look up ``fib[pop].kind``: a FORWARD
      entry advances to its next-hop sat, an EGRESS entry terminates
      the walk with the chosen GS id. Every ISL traversed plus the
      terminal sat feeder and GS feeder are charged to the
      :class:`UsageBook`.
    * **Measurement**: the FIB entry at the *ingress* already carries
      the full ``cost_ms`` from that sat down through the egress to
      the PoP, computed once upstream by
      :func:`precompute_per_sat_routing`. We only add the uplink RTT
      (user → ingress sat) here; ground RTT comes from
      ``ctx.ground_knowledge.estimator`` the same way the legacy
      forward does it.
    * **No throttling**: every flow that has a route is "served" at
      full demand. Max-min fair share / SLA degradation is deferred
      to the next phase.

Out of scope for this phase: satellite delay calibration (not yet
applied to routing-plane costs), service-class destinations (only
the geographic ground-truth estimator branch is exercised).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vantage.control.policy.common.utils import find_ingress_satellite
from vantage.domain import (
    CellGrid,
    EpochResult,
    FlowOutcome,
    NetworkSnapshot,
    RoutingPlane,
    TrafficDemand,
    UsageBook,
)

if TYPE_CHECKING:
    from vantage.engine.context import RunContext

__all__ = ["realize_via_routing_plane"]


def realize_via_routing_plane(
    routing_plane: RoutingPlane,
    cell_grid: CellGrid,
    usage_book: UsageBook,
    snapshot: NetworkSnapshot,
    demand: TrafficDemand,
    context: RunContext,
) -> EpochResult:
    """Execute one epoch's demand against a :class:`RoutingPlane`.

    For each flow:

        1. Resolve source cell via ``cell_grid``; resolve destination
           PoP via ``routing_plane.cell_to_pop``.
        2. Resolve the ingress satellite via user-side visibility
           (highest-elevation sat above the minimum elevation).
        3. Walk ``routing_plane`` hop-by-hop, starting from the
           ingress sat's FIB entry for the chosen PoP. Charge every
           ISL segment, the terminal sat's feeder, and the egress
           GS's feeder onto ``usage_book``.
        4. Compose ``satellite_rtt = uplink_rtt + ingress FIB cost``
           and ``ground_rtt`` from the ground truth estimator. Emit a
           :class:`FlowOutcome`.

    Flows whose source is unknown, whose cell has no PoP assignment,
    whose ingress satellite is unresolvable, or whose FIB walk fails
    to terminate at an EGRESS entry are dropped (counted as
    unrouted). No exception escapes.
    """
    sat = snapshot.satellite
    ground_truth = context.ground_knowledge.estimator
    outcomes: list[FlowOutcome] = []
    total_demand = 0.0
    routed_demand = 0.0

    # Bound FIB walks so a malformed plane can't spin forever.
    max_hops = sat.num_sats

    for flow_key, flow_demand in demand.flows.items():
        total_demand += flow_demand

        src_ep = context.endpoints.get(flow_key.src)
        if src_ep is None:
            continue

        # ── DECISION: cell → PoP lookup ───────────────────────────────
        try:
            cell_id = cell_grid.cell_of(flow_key.src)
        except KeyError:
            continue
        try:
            pop_code = routing_plane.cell_to_pop.pop_of(cell_id)
        except KeyError:
            continue

        # ── INGRESS: user → nearest visible sat ───────────────────────
        uplink = find_ingress_satellite(src_ep, sat.positions)
        if uplink is None:
            continue
        ingress = uplink.sat_id

        # The ingress FIB already encodes the full sat-side cost from
        # ``ingress`` down through the egress to the PoP (ISL + downlink
        # + backhaul). We only need the uplink RTT on top of it.
        try:
            ingress_fib = routing_plane.fib_of(ingress)
            ingress_entry = ingress_fib.route(pop_code)
        except KeyError:
            continue

        # ── PATH WALK: enumerate every sat on the route for bookkeeping ─
        # A correctly-built FIB from ``precompute_per_sat_routing`` is
        # always a shortest-path tree and cannot contain cycles. But a
        # hand-injected plane (tests, future controllers) could — so we
        # guard against arbitrary-length cycles with a ``visited`` set
        # rather than only catching direct self-loops. ``max_hops + 1``
        # serves as a belt-and-suspenders escape.
        path_sats: list[int] = [ingress]
        visited: set[int] = {ingress}
        egress_sat: int | None = None
        egress_gs: str | None = None
        current_sat = ingress
        for _ in range(max_hops + 1):
            try:
                entry = routing_plane.fib_of(current_sat).route(pop_code)
            except KeyError:
                break
            if entry.is_egress:
                egress_sat = current_sat
                egress_gs = entry.egress_gs
                break
            # FORWARD: advance, rejecting any cycle (including 2-cycles).
            next_hop = entry.next_hop_sat
            if next_hop in visited:
                break
            visited.add(next_hop)
            path_sats.append(next_hop)
            current_sat = next_hop
        if egress_sat is None or egress_gs is None:
            continue

        # ── CHARGE CAPACITY ──────────────────────────────────────────
        for i in range(len(path_sats) - 1):
            usage_book.charge_isl(path_sats[i], path_sats[i + 1], flow_demand)
        usage_book.charge_sat_feeder(egress_sat, flow_demand)
        usage_book.charge_gs_feeder(egress_gs, flow_demand)

        # ── MEASUREMENT: compose RTTs ────────────────────────────────
        # ``ingress_entry.cost_ms`` is the full sat-side RTT from the
        # ingress sat down to the PoP (computed once by
        # ``precompute_per_sat_routing``). The only thing it does *not*
        # include is the user-side uplink access delay, which we add
        # here as uplink one-way × 2.
        uplink_rtt = uplink.delay * 2
        satellite_rtt = uplink_rtt + ingress_entry.cost_ms

        # Strict measurement lookup — no geographic fallback. An
        # unknown (pop, dest) pair drops the flow (counted unrouted).
        if ground_truth is None:
            continue
        try:
            ground_rtt = ground_truth.estimate(pop_code, flow_key.dst) * 2
        except KeyError:
            continue

        total_rtt = satellite_rtt + ground_rtt

        outcomes.append(
            FlowOutcome(
                flow_key=flow_key,
                pop_code=pop_code,
                gs_id=egress_gs,
                user_sat=ingress,
                egress_sat=egress_sat,
                satellite_rtt=satellite_rtt,
                ground_rtt=ground_rtt,
                total_rtt=total_rtt,
                demand_gbps=flow_demand,
            )
        )
        routed_demand += flow_demand

    return EpochResult(
        epoch=snapshot.epoch,
        flow_outcomes=tuple(outcomes),
        total_demand_gbps=total_demand,
        routed_demand_gbps=routed_demand,
        unrouted_demand_gbps=total_demand - routed_demand,
    )
