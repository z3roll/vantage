"""Data plane: flow-level PoP selection + delay computation.

**RoutingPlaneForward**: controller pre-commits PoP assignments
via cell→PoP mapping and per-satellite FIBs, with capacity
tracking via :class:`UsageBook`.

Three-phase :class:`ForwardStrategy` (since the 2026-04-17 audit):

  1. ``decide``  — resolve cell→PoP, FIB walk, ground RTT. Pure read.
                   Returns ``None`` if any step fails — *nothing* is
                   charged in that case.
  2. ``charge``  — apply the flow's demand to the
                   :class:`UsageBook`.
  3. ``measure`` — read the *final* per-link load and compute
                   queuing/loss/bottleneck for the flow.

:func:`realize` runs the strategy in **two passes**: pass 1 decides
+ charges every flow, pass 2 measures each surviving flow against the
final per-link load. This decouples the per-flow queuing-delay
report from the iteration order of ``demand.flows``; pre-audit code
charged then measured inside the same per-flow loop, so flows landed
on each link with a load that depended on where they appeared in the
dict iteration.

Per-source ingress satellite is cached for the duration of one
:func:`realize` call: a terminal picks one serving sat and reuses it
for every flow it originates within that epoch, matching how a real
dish behaves rather than re-rolling the stochastic ``find_ingress_*``
per flow.

Produces :class:`EpochResult` output. All delays in ms.
"""

from __future__ import annotations

import random as _random
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
    "PathDecision",
    "PrecomputedPath",
    "ResolvedFlow",
    "RoutingPlaneForward",
    "effective_throughput",
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


@dataclass(frozen=True, slots=True)
class PathDecision:
    """Path-level decision for a flow — produced by ``decide``, consumed
    by ``charge`` and ``measure``.

    Carries every piece of information the latter two phases need so
    they don't have to repeat the cell→PoP / FIB-walk / ground-truth
    work. ``ground_rtt`` is resolved at decision time precisely so a
    failed ground lookup short-circuits *before* anything is charged.
    """

    pop_code: str
    gs_id: str
    user_sat: int
    egress_sat: int
    isl_links: tuple[tuple[int, int], ...]
    propagation_rtt: float  # uplink + sat-side path cost (RTT)
    ground_rtt: float       # PoP→destination (RTT)


class ForwardStrategy(Protocol):
    """Three-phase data plane.

    See module docstring for why this is split. Strategies are
    expected to be reusable across the per-flow loop within a single
    :func:`realize` call; they should not retain per-flow state.
    """

    def decide(
        self,
        flow_key: FlowKey,
        src_ep: Endpoint,
        ingress: int,
        uplink: AccessLink,
        snapshot: NetworkSnapshot,
        context: RunContext,
        epoch: int,
    ) -> PathDecision | None:
        """Resolve PoP, sat path, and ground RTT. Return ``None`` to
        mark the flow unrouted; callers will not invoke ``charge`` /
        ``measure`` for that flow."""
        ...

    def charge(self, decision: PathDecision, flow_demand: float) -> None:
        """Apply this flow's demand to the underlying :class:`UsageBook`.

        Called only when ``decide`` returned a non-``None``
        :class:`PathDecision`."""
        ...

    def measure(
        self, decision: PathDecision, snapshot: NetworkSnapshot,
    ) -> ResolvedFlow:
        """Compute per-link queuing/loss/bottleneck using the *current*
        :class:`UsageBook` state.

        :func:`realize` calls this in pass 2 after every flow has been
        charged, so every measurement reflects the steady-state load."""
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
    """Execute one epoch's demand through *strategy* in two passes.

    Pass 1 (decide + charge): for every flow with a known source and a
    visible satellite, ask the strategy for a :class:`PathDecision`.
    On success, charge the flow's demand to the strategy's
    :class:`UsageBook` and queue the decision for measurement. Flows
    with no decision are counted as unrouted and never touch the
    book.

    Pass 2 (measure): walk the queued decisions, ask the strategy to
    measure per-link queuing/loss/bottleneck against the *final*
    book state, and emit a :class:`FlowOutcome`. Because every flow
    is charged before any measurement happens, every flow on a given
    link sees the same steady-state utilisation — the per-flow
    queuing report no longer depends on dict iteration order.

    Each source's ingress satellite is resolved exactly once per
    :func:`realize` call (cached in ``_uplink_cache``). Without the
    cache, ``find_ingress_satellite``'s 80/20 stochastic branch (over
    a process-wide RNG) could scatter a single terminal's flows
    across multiple ingress sats within the same epoch.

    The stochastic branch uses a per-realize RNG seeded by the
    epoch number, so that ingress sat assignment is reproducible
    across runs with identical demand AND independent of any other
    module that might be sharing the global utils ``_RNG``.
    """
    sat = snapshot.satellite
    total_demand = 0.0

    _access = SphericalAccessModel()
    _visible_cache: dict[str, list[AccessLink]] = {}
    _uplink_cache: dict[str, AccessLink | None] = {}
    _ingress_rng = _random.Random(demand.epoch)

    pending: list[tuple[FlowKey, float, PathDecision]] = []

    # ── Pass 1: decide + charge ──
    for flow_key, flow_demand in demand.flows.items():
        total_demand += flow_demand

        src_ep = context.endpoints.get(flow_key.src)
        if src_ep is None:
            continue

        src_name = flow_key.src
        if src_name not in _uplink_cache:
            if src_name not in _visible_cache:
                _visible_cache[src_name] = _access.compute_access(
                    src_ep.lat_deg, src_ep.lon_deg, 0.0, sat.positions,
                    DEFAULT_MIN_ELEVATION_DEG,
                )
            visible = _visible_cache[src_name]
            _uplink_cache[src_name] = (
                find_ingress_satellite(
                    src_ep, sat.positions,
                    rng=_ingress_rng, _visible=visible,
                )
                if visible else None
            )
        uplink = _uplink_cache[src_name]
        if uplink is None:
            continue

        decision = strategy.decide(
            flow_key, src_ep, uplink.sat_id, uplink,
            snapshot, context, demand.epoch,
        )
        if decision is None:
            continue

        strategy.charge(decision, flow_demand)
        pending.append((flow_key, flow_demand, decision))

    # ── Pass 2: measure with final loads + emit outcomes ──
    outcomes: list[FlowOutcome] = []
    routed_demand = 0.0
    for flow_key, flow_demand, decision in pending:
        resolved = strategy.measure(decision, snapshot)
        total_rtt = resolved.satellite_rtt + resolved.ground_rtt
        eff_tput = effective_throughput(
            flow_demand, total_rtt,
            resolved.loss_probability, resolved.bottleneck_gbps,
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


def effective_throughput(
    demand_gbps: float,
    total_rtt_ms: float,
    loss_probability: float,
    bottleneck_gbps: float,
) -> float:
    """Cap demand by the strictest of {requested, PFTK, bottleneck}.

    Earlier code's loss-branch returned ``min(demand, pftk)`` and
    silently ignored ``bottleneck_gbps``. PFTK at low-but-nonzero
    loss can far exceed the physical bottleneck (it is window-limited
    by ``DEFAULT_MAX_WINDOW_BYTES`` rather than by the link), so
    effective throughput was reported above what the link could
    actually carry.
    """
    candidates = [demand_gbps]
    if loss_probability > 0 and total_rtt_ms > 0:
        candidates.append(pftk_throughput(total_rtt_ms, loss_probability))
    if bottleneck_gbps > 0:
        candidates.append(bottleneck_gbps)
    return min(candidates)


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

    def decide(
        self,
        flow_key: FlowKey,
        src_ep: Endpoint,
        ingress: int,
        uplink: AccessLink,
        snapshot: NetworkSnapshot,
        context: RunContext,
        epoch: int,
    ) -> PathDecision | None:
        del src_ep, epoch  # unused — present for Protocol signature stability
        sat = snapshot.satellite

        # ── cell → PoP ──
        try:
            cell_id = self._grid.cell_of(flow_key.src)
        except KeyError:
            return None
        try:
            pop_code = self._plane.cell_to_pop.pop_of(cell_id, dest=flow_key.dst)
        except KeyError:
            return None

        # ── path: table lookup, fallback to live FIB walk ──
        ppath = self._paths.get((ingress, pop_code))
        if ppath is None:
            ppath = self._fib_walk(ingress, pop_code, sat.num_sats)
        if ppath is None:
            return None

        # ── ground RTT — resolved here so a missing measurement
        # short-circuits the flow before any capacity is charged. ──
        ground_truth = context.ground_knowledge.estimator
        if ground_truth is None:
            return None
        try:
            ground_rtt = ground_truth.estimate(pop_code, flow_key.dst) * 2
        except KeyError:
            return None

        return PathDecision(
            pop_code=pop_code,
            gs_id=ppath.egress_gs,
            user_sat=ingress,
            egress_sat=ppath.egress_sat,
            isl_links=ppath.isl_links,
            propagation_rtt=uplink.delay * 2 + ppath.cost_ms,
            ground_rtt=ground_rtt,
        )

    def charge(self, decision: PathDecision, flow_demand: float) -> None:
        for a, b in decision.isl_links:
            self._book.charge_isl(a, b, flow_demand)
        self._book.charge_sat_feeder(decision.egress_sat, flow_demand)
        self._book.charge_gs_feeder(decision.gs_id, flow_demand)

    def measure(
        self, decision: PathDecision, snapshot: NetworkSnapshot,
    ) -> ResolvedFlow:
        sat = snapshot.satellite

        hop_losses: list[float] = []
        hop_capacities: list[float] = []
        total_queuing_oneway = 0.0
        total_tx_oneway = 0.0

        for a, b in decision.isl_links:
            isl_prop = float(sat.delay_matrix[a, b])
            isl_cap = self._book.view.isl_cap(a, b)
            isl_load = self._book.isl_used.get(self._book.isl_key(a, b), 0.0)
            perf = link_performance(isl_prop, isl_cap, isl_load)
            total_queuing_oneway += perf.queuing_ms
            total_tx_oneway += perf.transmission_ms
            hop_losses.append(perf.loss_probability)
            hop_capacities.append(isl_cap)

        sf_cap = self._book.view.sat_feeder_cap(decision.egress_sat)
        sf_load = self._book.sat_feeder_used.get(decision.egress_sat, 0.0)
        sf_perf = link_performance(0.0, sf_cap, sf_load)
        total_queuing_oneway += sf_perf.queuing_ms
        total_tx_oneway += sf_perf.transmission_ms
        hop_losses.append(sf_perf.loss_probability)
        hop_capacities.append(sf_cap)

        gf_cap = self._book.view.gs_feeder_cap(decision.gs_id)
        gf_load = self._book.gs_feeder_used.get(decision.gs_id, 0.0)
        gf_perf = link_performance(0.0, gf_cap, gf_load)
        total_queuing_oneway += gf_perf.queuing_ms
        total_tx_oneway += gf_perf.transmission_ms
        hop_losses.append(gf_perf.loss_probability)
        hop_capacities.append(gf_cap)

        queuing_rtt = total_queuing_oneway * 2
        transmission_rtt = total_tx_oneway * 2
        satellite_rtt = decision.propagation_rtt + queuing_rtt + transmission_rtt

        return ResolvedFlow(
            pop_code=decision.pop_code,
            gs_id=decision.gs_id,
            user_sat=decision.user_sat,
            egress_sat=decision.egress_sat,
            satellite_rtt=satellite_rtt,
            ground_rtt=decision.ground_rtt,
            propagation_rtt=decision.propagation_rtt,
            queuing_rtt=queuing_rtt,
            transmission_rtt=transmission_rtt,
            loss_probability=path_loss(hop_losses),
            bottleneck_gbps=bottleneck_capacity(hop_capacities),
        )
