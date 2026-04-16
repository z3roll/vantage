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

import math
import random as _random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

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
    "EgressOption",
    "ForwardStrategy",
    "PathDecision",
    "ResolvedFlow",
    "RoutingPlaneForward",
    "compute_egress_options",
    "effective_throughput",
    "realize",
]


# ---------------------------------------------------------------------------
# Egress options + helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EgressOption:
    """One ``(egress_sat, gs, path)`` candidate for routing a flow.

    Each option fully describes a downlink path: which PoP it lands
    at (might be the controller's chosen PoP or the cell's baseline
    fallback PoP — the two have different ``ground_rtt``), the
    egress sat that downlinks to ``gs_id``, the ISL hops from the
    flow's ingress to that egress, and the propagation/ground RTTs.

    Used by the data-plane multi-egress reroute logic: when the
    primary option's egress sat-feeder has no remaining capacity,
    :meth:`RoutingPlaneForward.charge` walks alternates in order
    and uses the first one with room.
    """

    pop_code: str
    egress_sat: int
    gs_id: str
    isl_links: tuple[tuple[int, int], ...]
    propagation_rtt: float    # uplink + sat-segment (RTT)
    ground_rtt: float         # PoP→destination (RTT)


def compute_egress_options(
    snapshot: NetworkSnapshot,
    ingress: int,
    pop_code: str,
    k: int,
) -> tuple[EgressOption, ...]:
    """Top-K (egress_sat, gs, path) options to reach ``pop_code`` from
    ``ingress``, ranked ascending by sat-segment RTT.

    Enumerates every ``(gs, sat)`` pair such that ``gs`` is attached
    to ``pop_code`` and ``sat`` is a visible egress for ``gs``,
    computes the round-trip cost ``ISL + downlink + backhaul``, sorts
    ascending, takes the top *K*. Each returned option's
    ``propagation_rtt`` carries only the *sat segment* — the caller
    adds the uplink RTT before reporting the user-facing latency.
    ``ground_rtt`` is set to 0 here; the caller fills it from the
    ground-delay estimator (which depends on the destination, not
    the route).

    Used by :meth:`RoutingPlaneForward.decide` to enumerate
    alternates for the per-sat-feeder reroute path. Cached per
    ``(ingress, pop)`` inside ``RoutingPlaneForward`` because the
    work is identical across all flows that share that pair within
    one epoch.
    """
    sat = snapshot.satellite
    infra = snapshot.infra
    candidates: list[tuple[float, int, str]] = []

    for gs_id, backhaul_oneway in infra.pop_gs_edges(pop_code):
        if infra.gs_by_id(gs_id) is None:
            continue
        gs_links = sat.gateway_attachments.attachments.get(gs_id)
        if not gs_links:
            continue
        backhaul_rtt = backhaul_oneway * 2
        for link in gs_links:
            egress = link.sat_id
            isl_one = float(sat.delay_matrix[ingress, egress])
            if not math.isfinite(isl_one):
                continue
            cost = isl_one * 2 + link.delay * 2 + backhaul_rtt
            candidates.append((cost, egress, gs_id))

    candidates.sort()

    options: list[EgressOption] = []
    for cost, egress, gs_id in candidates[:k]:
        if egress == ingress:
            isl_links: tuple[tuple[int, int], ...] = ()
        else:
            walked = _walk_isl_path(sat.predecessor_matrix, ingress, egress)
            if walked is None:
                continue
            isl_links = walked
        options.append(EgressOption(
            pop_code=pop_code,
            egress_sat=egress,
            gs_id=gs_id,
            isl_links=isl_links,
            # ISL + downlink + backhaul RTT; the caller (decide) adds
            # the uplink RTT to produce the user-facing propagation
            # budget. Ground RTT is per-PoP×destination so it lives
            # outside this helper.
            propagation_rtt=cost,
            ground_rtt=0.0,
        ))
    return tuple(options)


def _walk_isl_path(
    pred: NDArray[np.int32], src: int, dst: int,
) -> tuple[tuple[int, int], ...] | None:
    """Reconstruct the ISL hop sequence from ``src`` to ``dst`` using
    the shortest-path predecessor matrix.

    ``pred[s, t]`` is "predecessor of t on the shortest path from s",
    so we walk backwards from ``dst`` collecting nodes until we hit
    ``src``. Returns ``None`` if any predecessor along the way is
    ``-1`` (unreachable) or if the walk fails to terminate within
    ``n_sats`` iterations (pathological input). For ``src == dst``
    returns an empty tuple (no ISL hops needed)."""
    if src == dst:
        return ()
    rev_path: list[int] = [dst]
    cur = dst
    n = int(pred.shape[0])
    for _ in range(n):
        prev = int(pred[src, cur])
        if prev < 0:
            return None
        rev_path.append(prev)
        if prev == src:
            break
        cur = prev
    else:
        return None
    path = list(reversed(rev_path))
    return tuple((path[i], path[i + 1]) for i in range(len(path) - 1))


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
    """Path-level routing decision for a flow.

    Carries a ranked list of :class:`EgressOption` to try in order:

    * ``options[0]`` — primary (best E2E cost) for the controller's
      chosen PoP.
    * ``options[1..K-1]`` — alternates for the same PoP, ranked
      ascending by E2E cost. Used when the primary's egress sat
      feeder (20 Gbps) is saturated.
    * ``options[-1]`` (if it lives at a different ``pop_code`` than
      the rest) — baseline-PoP fallback, the cell's geographic
      nearest PoP. Used regardless of capacity if every preceding
      option is saturated, mirroring the controller-side overflow
      semantics.

    :meth:`RoutingPlaneForward.charge` walks the list in order,
    picks the first egress sat with remaining feeder cap, and
    returns the chosen option so :meth:`measure` can report the
    actual user-facing path metrics.

    ``options`` must be non-empty: ``RoutingPlaneForward.decide``
    returns ``None`` (rather than an empty-options ``PathDecision``)
    when no candidate egress is reachable, so by the time a
    ``PathDecision`` exists ``charge``'s ``options[-1]`` indexing
    is safe. The invariant is enforced in ``__post_init__``.
    """

    user_sat: int
    options: tuple[EgressOption, ...]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(
                "PathDecision.options must be non-empty; callers should "
                "return None from decide() instead of constructing an "
                "empty-options decision",
            )


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
        """Resolve PoP, ranked egress options, and per-PoP ground
        RTTs. Return ``None`` to mark the flow unrouted; callers
        will not invoke ``charge`` / ``measure`` for that flow."""
        ...

    def charge(
        self, decision: PathDecision, flow_demand: float,
    ) -> EgressOption:
        """Pick an option from ``decision`` (primary if it has cap,
        else alternate, else fallback) and apply ``flow_demand`` to
        the underlying :class:`UsageBook`. Return the chosen option
        so :meth:`measure` can use it in pass 2.

        Called only when ``decide`` returned a non-``None``
        :class:`PathDecision`."""
        ...

    def measure(
        self, decision: PathDecision, chosen: EgressOption,
        snapshot: NetworkSnapshot,
    ) -> ResolvedFlow:
        """Compute per-link queuing/loss/bottleneck for ``chosen``
        using the *current* :class:`UsageBook` state.

        :func:`realize` calls this in pass 2 after every flow has
        been charged, so every measurement reflects the steady-state
        load."""
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

    pending: list[tuple[FlowKey, float, PathDecision, EgressOption]] = []

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

        chosen = strategy.charge(decision, flow_demand)
        pending.append((flow_key, flow_demand, decision, chosen))

    # ── Pass 2: measure with final loads + emit outcomes ──
    outcomes: list[FlowOutcome] = []
    routed_demand = 0.0
    for flow_key, flow_demand, decision, chosen in pending:
        resolved = strategy.measure(decision, chosen, snapshot)
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
    """Controller-committed PoP selection with multi-egress reroute.

    Per-flow phases:

    1. ``decide`` — cell→PoP via ``RoutingPlane.cell_to_pop`` (with
       optional per-dest override), then enumerate the top-K
       ``EgressOption`` candidates for that ``(ingress, pop)`` and
       append a fallback option that lands at the cell's geographic
       nearest PoP. Resolves ground RTT for both PoPs eagerly so a
       missing measurement short-circuits before any charge.

    2. ``charge`` — try the options in rank order; pick the first
       whose ``egress_sat`` has remaining sat-feeder capacity (20
       Gbps per Ka antenna). If every option's egress sat is
       saturated, take the *fallback* option regardless of capacity
       (overflow accepted; surfaces in measure as high queuing/loss).

    3. ``measure`` — use the chosen option's path metrics + the
       final :class:`UsageBook` state to compute per-link
       queuing/loss/bottleneck.

    The ``(ingress, pop) → tuple[EgressOption, ...]`` cache is
    populated lazily on first use within a ``RoutingPlaneForward``
    instance and naturally bounded — the engine constructs a fresh
    instance per epoch, so no cross-epoch leakage.
    """

    __slots__ = ("_book", "_grid", "_k", "_options_cache", "_plane")

    def __init__(
        self,
        routing_plane: RoutingPlane,
        cell_grid: CellGrid,
        usage_book: UsageBook,
        path_table: object | None = None,    # ignored — kept for callers
        k: int = 8,
    ) -> None:
        del path_table  # legacy positional arg from the pre-multi-egress API
        self._plane = routing_plane
        self._grid = cell_grid
        self._book = usage_book
        self._k = k
        self._options_cache: dict[tuple[int, str], tuple[EgressOption, ...]] = {}

    def _options_for(
        self, ingress: int, pop_code: str, snapshot: NetworkSnapshot,
    ) -> tuple[EgressOption, ...]:
        key = (ingress, pop_code)
        cached = self._options_cache.get(key)
        if cached is None:
            cached = compute_egress_options(snapshot, ingress, pop_code, self._k)
            self._options_cache[key] = cached
        return cached

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

        # ── cell → controller-chosen PoP ──
        try:
            cell_id = self._grid.cell_of(flow_key.src)
        except KeyError:
            return None
        try:
            pop_code = self._plane.cell_to_pop.pop_of(cell_id, dest=flow_key.dst)
        except KeyError:
            return None

        ground_truth = context.ground_knowledge.estimator
        if ground_truth is None:
            return None
        try:
            ground_rtt_chosen = ground_truth.estimate(pop_code, flow_key.dst) * 2
        except KeyError:
            return None

        chosen_options = self._options_for(ingress, pop_code, snapshot)
        if not chosen_options:
            return None

        uplink_rtt = uplink.delay * 2
        # Lift each helper-emitted option into a "uplink + ground"-aware
        # EgressOption usable by measure() directly.
        primary_options: tuple[EgressOption, ...] = tuple(
            EgressOption(
                pop_code=opt.pop_code,
                egress_sat=opt.egress_sat,
                gs_id=opt.gs_id,
                isl_links=opt.isl_links,
                propagation_rtt=uplink_rtt + opt.propagation_rtt,
                ground_rtt=ground_rtt_chosen,
            )
            for opt in chosen_options
        )

        # ── Append baseline-PoP fallback if it differs from the
        # chosen PoP. The fallback is the controller's own overflow
        # semantic mirrored into the data plane: if every primary
        # option is saturated, fall back to the cell's geographic
        # nearest PoP. ──
        baseline_pop = self._plane.cell_to_pop.mapping[cell_id]
        if baseline_pop == pop_code:
            return PathDecision(user_sat=ingress, options=primary_options)

        try:
            ground_rtt_baseline = ground_truth.estimate(
                baseline_pop, flow_key.dst,
            ) * 2
        except KeyError:
            # Baseline ground unmeasured — no useful fallback to add.
            return PathDecision(user_sat=ingress, options=primary_options)

        baseline_options = self._options_for(ingress, baseline_pop, snapshot)
        if not baseline_options:
            return PathDecision(user_sat=ingress, options=primary_options)

        bp = baseline_options[0]
        fallback = EgressOption(
            pop_code=bp.pop_code,
            egress_sat=bp.egress_sat,
            gs_id=bp.gs_id,
            isl_links=bp.isl_links,
            propagation_rtt=uplink_rtt + bp.propagation_rtt,
            ground_rtt=ground_rtt_baseline,
        )
        return PathDecision(
            user_sat=ingress, options=primary_options + (fallback,),
        )

    def charge(
        self, decision: PathDecision, flow_demand: float,
    ) -> EgressOption:
        # Per-sat Ka feeder cap; constant 20 Gbps for the default shell
        # but read via CapacityView for forward-compat with multi-shell
        # deployments.
        for option in decision.options:
            cap = self._book.view.sat_feeder_cap(option.egress_sat)
            current = self._book.sat_feeder_used.get(option.egress_sat, 0.0)
            if current + flow_demand <= cap:
                self._do_charge(option, flow_demand)
                return option

        # Every option's egress sat is saturated. Take the fallback
        # (= last option) and accept the overflow — measure() will
        # report it as elevated queuing/loss rather than dropping the
        # flow.
        chosen = decision.options[-1]
        self._do_charge(chosen, flow_demand)
        return chosen

    def _do_charge(self, option: EgressOption, demand: float) -> None:
        for a, b in option.isl_links:
            self._book.charge_isl(a, b, demand)
        self._book.charge_sat_feeder(option.egress_sat, demand)
        self._book.charge_gs_feeder(option.gs_id, demand)

    def measure(
        self,
        decision: PathDecision,
        chosen: EgressOption,
        snapshot: NetworkSnapshot,
    ) -> ResolvedFlow:
        sat = snapshot.satellite

        hop_losses: list[float] = []
        hop_capacities: list[float] = []
        total_queuing_oneway = 0.0
        total_tx_oneway = 0.0

        for a, b in chosen.isl_links:
            isl_prop = float(sat.delay_matrix[a, b])
            isl_cap = self._book.view.isl_cap(a, b)
            isl_load = self._book.isl_used.get(self._book.isl_key(a, b), 0.0)
            perf = link_performance(isl_prop, isl_cap, isl_load)
            total_queuing_oneway += perf.queuing_ms
            total_tx_oneway += perf.transmission_ms
            hop_losses.append(perf.loss_probability)
            hop_capacities.append(isl_cap)

        sf_cap = self._book.view.sat_feeder_cap(chosen.egress_sat)
        sf_load = self._book.sat_feeder_used.get(chosen.egress_sat, 0.0)
        sf_perf = link_performance(0.0, sf_cap, sf_load)
        total_queuing_oneway += sf_perf.queuing_ms
        total_tx_oneway += sf_perf.transmission_ms
        hop_losses.append(sf_perf.loss_probability)
        hop_capacities.append(sf_cap)

        gf_cap = self._book.view.gs_feeder_cap(chosen.gs_id)
        gf_load = self._book.gs_feeder_used.get(chosen.gs_id, 0.0)
        gf_perf = link_performance(0.0, gf_cap, gf_load)
        total_queuing_oneway += gf_perf.queuing_ms
        total_tx_oneway += gf_perf.transmission_ms
        hop_losses.append(gf_perf.loss_probability)
        hop_capacities.append(gf_cap)

        queuing_rtt = total_queuing_oneway * 2
        transmission_rtt = total_tx_oneway * 2
        satellite_rtt = chosen.propagation_rtt + queuing_rtt + transmission_rtt

        return ResolvedFlow(
            pop_code=chosen.pop_code,
            gs_id=chosen.gs_id,
            user_sat=decision.user_sat,
            egress_sat=chosen.egress_sat,
            satellite_rtt=satellite_rtt,
            ground_rtt=chosen.ground_rtt,
            propagation_rtt=chosen.propagation_rtt,
            queuing_rtt=queuing_rtt,
            transmission_rtt=transmission_rtt,
            loss_probability=path_loss(hop_losses),
            bottleneck_gbps=bottleneck_capacity(hop_capacities),
        )
