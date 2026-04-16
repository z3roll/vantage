"""Data plane: flow-level PoP selection + delay computation.

**RoutingPlaneForward**: controller pre-commits PoP assignments
via cell→PoP mapping and per-satellite FIBs, with capacity
tracking via :class:`UsageBook`.

Produces :class:`EpochResult` output. All delays in ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vantage.common import DEFAULT_MIN_ELEVATION_DEG
from vantage.common.link_model import (
    bottleneck_capacity,
    link_performance,
    path_loss,
    pftk_throughput,
)
from vantage.control.policy.common.utils import find_ingress_satellite
from vantage.domain import (
    AccessLink,
    CellGrid,
    Endpoint,
    EpochResult,
    FlowKey,
    FlowOutcome,
    NetworkSnapshot,
    RoutingPlane,
    TrafficDemand,
    UsageBook,
)
from vantage.world.satellite.visibility import SphericalAccessModel

if TYPE_CHECKING:
    from vantage.engine.context import RunContext

__all__ = [
    "ForwardStrategy",
    "PrecomputedPath",
    "ResolvedFlow",
    "RoutingPlaneForward",
    "precompute_path_table",
    "realize",
]


# ---------------------------------------------------------------------------
# Pre-computed path table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrecomputedPath:
    """Cached result of a FIB walk from one satellite to one PoP."""

    egress_sat: int
    egress_gs: str
    isl_links: tuple[tuple[int, int], ...]
    cost_ms: float  # ingress FIB entry cost_ms


def precompute_path_table(
    plane: RoutingPlane,
    num_sats: int,
) -> dict[tuple[int, str], PrecomputedPath]:
    """Walk every (satellite, PoP) FIB entry once, cache the result.

    Called when the RoutingPlane is rebuilt (~every 15s). The returned
    table replaces per-flow FIB walks in ``RoutingPlaneForward``.
    """
    table: dict[tuple[int, str], PrecomputedPath] = {}

    for sat_id in plane.sat_fibs:
        sat_fib = plane.sat_fibs[sat_id]
        for pop_code, entry in sat_fib.fib.items():
            path_sats: list[int] = [sat_id]
            visited: set[int] = {sat_id}
            egress_sat: int | None = None
            egress_gs: str | None = None
            current = sat_id

            for _ in range(num_sats):
                try:
                    e = plane.fib_of(current).route(pop_code)
                except KeyError:
                    break
                if e.is_egress:
                    egress_sat = current
                    egress_gs = e.egress_gs
                    break
                nxt = e.next_hop_sat
                if nxt in visited:
                    break
                visited.add(nxt)
                path_sats.append(nxt)
                current = nxt

            if egress_sat is not None and egress_gs is not None:
                isl_links = tuple(
                    (path_sats[i], path_sats[i + 1])
                    for i in range(len(path_sats) - 1)
                )
                table[(sat_id, pop_code)] = PrecomputedPath(
                    egress_sat=egress_sat,
                    egress_gs=egress_gs,
                    isl_links=isl_links,
                    cost_ms=entry.cost_ms,
                )

    return table


# ---------------------------------------------------------------------------
# Strategy protocol + shared result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedFlow:
    """Outcome of resolving a single flow through a forward strategy."""

    pop_code: str
    gs_id: str
    user_sat: int
    egress_sat: int
    satellite_rtt: float
    ground_rtt: float
    propagation_rtt: float = 0.0
    queuing_rtt: float = 0.0
    transmission_rtt: float = 0.0
    loss_probability: float = 0.0
    bottleneck_gbps: float = 0.0


class ForwardStrategy(Protocol):
    """Protocol for per-flow PoP selection, path resolution, and RTT computation."""

    def resolve_flow(
        self,
        flow_key: FlowKey,
        flow_demand: float,
        src_ep: Endpoint,
        ingress: int,
        uplink: AccessLink,
        snapshot: NetworkSnapshot,
        context: RunContext,
        epoch: int,
    ) -> ResolvedFlow | None:
        """Resolve a single flow.  Return *None* to drop (unrouted)."""
        ...


# ---------------------------------------------------------------------------
# Unified epoch loop
# ---------------------------------------------------------------------------


def realize(
    strategy: ForwardStrategy,
    snapshot: NetworkSnapshot,
    demand: TrafficDemand,
    context: RunContext,
) -> EpochResult:
    """Execute one epoch's demand through *strategy*.

    For each flow the strategy selects a PoP, resolves the satellite
    path, and computes satellite + ground RTTs.  Flows for which the
    strategy returns ``None`` are counted as unrouted.
    """
    sat = snapshot.satellite
    outcomes: list[FlowOutcome] = []
    total_demand = 0.0
    routed_demand = 0.0

    _access = SphericalAccessModel()
    _visible_cache: dict[str, list[AccessLink]] = {}

    for flow_key, flow_demand in demand.flows.items():
        total_demand += flow_demand

        src_ep = context.endpoints.get(flow_key.src)
        if src_ep is None:
            continue

        src_name = flow_key.src
        if src_name not in _visible_cache:
            _visible_cache[src_name] = _access.compute_access(
                src_ep.lat_deg, src_ep.lon_deg, 0.0, sat.positions,
                DEFAULT_MIN_ELEVATION_DEG,
            )
        visible = _visible_cache[src_name]
        if not visible:
            continue
        uplink = find_ingress_satellite(src_ep, sat.positions, _visible=visible)
        if uplink is None:
            continue

        resolved = strategy.resolve_flow(
            flow_key, flow_demand, src_ep, uplink.sat_id, uplink,
            snapshot, context, demand.epoch,
        )
        if resolved is None:
            continue

        total_rtt = resolved.satellite_rtt + resolved.ground_rtt
        if resolved.loss_probability > 0 and total_rtt > 0:
            tcp_gbps = pftk_throughput(total_rtt, resolved.loss_probability)
            eff_tput = min(flow_demand, tcp_gbps)
        else:
            eff_tput = (
                min(flow_demand, resolved.bottleneck_gbps)
                if resolved.bottleneck_gbps > 0
                else flow_demand
            )

        outcomes.append(FlowOutcome(
            flow_key=flow_key,
            pop_code=resolved.pop_code,
            gs_id=resolved.gs_id,
            user_sat=resolved.user_sat,
            egress_sat=resolved.egress_sat,
            satellite_rtt=resolved.satellite_rtt,
            ground_rtt=resolved.ground_rtt,
            total_rtt=total_rtt,
            demand_gbps=flow_demand,
            propagation_rtt=resolved.propagation_rtt,
            queuing_rtt=resolved.queuing_rtt,
            transmission_rtt=resolved.transmission_rtt,
            loss_probability=resolved.loss_probability,
            bottleneck_gbps=resolved.bottleneck_gbps,
            effective_throughput_gbps=eff_tput,
        ))
        routed_demand += flow_demand

    return EpochResult(
        epoch=demand.epoch,
        flow_outcomes=tuple(outcomes),
        total_demand_gbps=total_demand,
        routed_demand_gbps=routed_demand,
        unrouted_demand_gbps=total_demand - routed_demand,
    )


# ---------------------------------------------------------------------------
# RoutingPlane + FIB walk
# ---------------------------------------------------------------------------


class RoutingPlaneForward:
    """Controller-committed PoP selection with pre-computed path table.

    Uses :class:`PrecomputedPath` table (built once per RoutingPlane)
    instead of per-flow FIB walks.  Capacity charging and queuing
    delay are still computed per-flow at realize time.
    """

    __slots__ = ("_grid", "_plane", "_book", "_paths")

    def __init__(
        self,
        routing_plane: RoutingPlane,
        cell_grid: CellGrid,
        usage_book: UsageBook,
        path_table: dict[tuple[int, str], PrecomputedPath] | None = None,
    ) -> None:
        self._plane = routing_plane
        self._grid = cell_grid
        self._book = usage_book
        self._paths = path_table or {}

    def _fib_walk(self, ingress: int, pop_code: str, max_hops: int) -> PrecomputedPath | None:
        """Live FIB walk fallback when path_table misses."""
        try:
            entry = self._plane.fib_of(ingress).route(pop_code)
        except KeyError:
            return None
        path_sats = [ingress]
        visited = {ingress}
        current = ingress
        for _ in range(max_hops):
            try:
                e = self._plane.fib_of(current).route(pop_code)
            except KeyError:
                return None
            if e.is_egress:
                return PrecomputedPath(
                    egress_sat=current,
                    egress_gs=e.egress_gs or "",
                    isl_links=tuple(
                        (path_sats[i], path_sats[i + 1])
                        for i in range(len(path_sats) - 1)
                    ),
                    cost_ms=entry.cost_ms,
                )
            nxt = e.next_hop_sat
            if nxt in visited:
                return None
            visited.add(nxt)
            path_sats.append(nxt)
            current = nxt
        return None

    def resolve_flow(
        self,
        flow_key: FlowKey,
        flow_demand: float,
        src_ep: Endpoint,
        ingress: int,
        uplink: AccessLink,
        snapshot: NetworkSnapshot,
        context: RunContext,
        epoch: int,
    ) -> ResolvedFlow | None:
        sat = snapshot.satellite
        ground_truth = context.ground_knowledge.estimator

        # ── DECISION: cell → PoP ─────────────────────────────────────
        try:
            cell_id = self._grid.cell_of(flow_key.src)
        except KeyError:
            return None
        try:
            pop_code = self._plane.cell_to_pop.pop_of(cell_id, dest=flow_key.dst)
        except KeyError:
            return None

        # ── PATH: table lookup, fallback to FIB walk ────────────────
        ppath = self._paths.get((ingress, pop_code))
        if ppath is None:
            ppath = self._fib_walk(ingress, pop_code, sat.num_sats)
        if ppath is None:
            return None

        egress_sat = ppath.egress_sat
        egress_gs = ppath.egress_gs

        # ── CHARGE CAPACITY ──────────────────────────────────────────
        for a, b in ppath.isl_links:
            self._book.charge_isl(a, b, flow_demand)
        self._book.charge_sat_feeder(egress_sat, flow_demand)
        self._book.charge_gs_feeder(egress_gs, flow_demand)

        # ── MEASUREMENT: propagation baseline ────────────────────────
        uplink_rtt = uplink.delay * 2
        propagation_rtt = uplink_rtt + ppath.cost_ms

        # ── QUEUING + LOSS: per-hop ──────────────────────────────────
        hop_losses: list[float] = []
        hop_capacities: list[float] = []
        total_queuing_oneway = 0.0
        total_tx_oneway = 0.0

        for a, b in ppath.isl_links:
            isl_prop = float(sat.delay_matrix[a, b])
            isl_cap = self._book.view.isl_cap(a, b)
            isl_load = self._book.isl_used.get(self._book.isl_key(a, b), 0.0)
            perf = link_performance(isl_prop, isl_cap, isl_load)
            total_queuing_oneway += perf.queuing_ms
            total_tx_oneway += perf.transmission_ms
            hop_losses.append(perf.loss_probability)
            hop_capacities.append(isl_cap)

        sf_cap = self._book.view.sat_feeder_cap(egress_sat)
        sf_load = self._book.sat_feeder_used.get(egress_sat, 0.0)
        sf_perf = link_performance(0.0, sf_cap, sf_load)
        total_queuing_oneway += sf_perf.queuing_ms
        total_tx_oneway += sf_perf.transmission_ms
        hop_losses.append(sf_perf.loss_probability)
        hop_capacities.append(sf_cap)

        gf_cap = self._book.view.gs_feeder_cap(egress_gs)
        gf_load = self._book.gs_feeder_used.get(egress_gs, 0.0)
        gf_perf = link_performance(0.0, gf_cap, gf_load)
        total_queuing_oneway += gf_perf.queuing_ms
        total_tx_oneway += gf_perf.transmission_ms
        hop_losses.append(gf_perf.loss_probability)
        hop_capacities.append(gf_cap)

        queuing_rtt = total_queuing_oneway * 2
        transmission_rtt = total_tx_oneway * 2
        satellite_rtt = propagation_rtt + queuing_rtt + transmission_rtt
        e2e_loss = path_loss(hop_losses)
        bneck = bottleneck_capacity(hop_capacities)

        if ground_truth is None:
            return None
        try:
            ground_rtt = ground_truth.estimate(pop_code, flow_key.dst) * 2
        except KeyError:
            return None

        return ResolvedFlow(
            pop_code=pop_code,
            gs_id=egress_gs,
            user_sat=ingress,
            egress_sat=egress_sat,
            satellite_rtt=satellite_rtt,
            ground_rtt=ground_rtt,
            propagation_rtt=propagation_rtt,
            queuing_rtt=queuing_rtt,
            transmission_rtt=transmission_rtt,
            loss_probability=e2e_loss,
            bottleneck_gbps=bneck,
        )
