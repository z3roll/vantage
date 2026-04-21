"""Data plane: flow-level PoP selection + delay computation.

**RoutingPlaneForward**: controller pre-commits a per-(cell, dest)
ranked PoP cascade and per-satellite FIBs; the data plane walks
the cascade × top-K sats per PoP and picks the first feasible
egress against the per-Ka-feeder capacity tracked in
:class:`UsageBook`.

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

import logging
import math
import random as _random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from vantage.common import DEFAULT_MIN_ELEVATION_DEG
from vantage.common.link_model import (
    LinkPerformance,
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


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Egress options + helper
# ---------------------------------------------------------------------------


class EgressOption(NamedTuple):
    """One ``(egress_sat, gs, path)`` candidate for routing a flow.

    Each option fully describes a downlink path: which PoP it lands
    at (one of the PoPs in the cell's ranked cascade — different PoPs
    have different ``ground_rtt`` to the same destination), the
    egress sat that downlinks to ``gs_id``, the ISL hops from the
    flow's ingress to that egress, and the propagation/ground RTTs.

    Used by the data-plane multi-egress reroute logic: when one
    option's egress sat feeder has no remaining capacity,
    :meth:`RoutingPlaneForward.charge` walks alternates in order
    (within the same PoP first, then across the controller's
    next-ranked PoP) and uses the first one with room.

    Implemented as ``NamedTuple`` (not ``@dataclass(frozen=True,
    slots=True)``) because decide() allocates up to
    ``max_cascade_pops × k`` of these per flow: ``NamedTuple.__new__``
    is ~3× faster than the frozen-dataclass constructor at our scale
    (1.9 M allocations/epoch), and the tuple backing also lets
    ``min(options, key=...)`` iterate without attribute lookups.
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
    path_walker: "Callable[[int, int], tuple[tuple[int, int], ...] | None] | None" = None,
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

    ``path_walker`` lets the caller supply a memoized ``_walk_isl_path``
    — the same ``(ingress, egress)`` pair is walked by every ``(pop)``
    whose attached GSs include that egress sat, so per-realize caching
    collapses tens of thousands of walk calls into the ~unique-sat
    count. If ``None``, falls back to calling ``_walk_isl_path``
    directly (no memoization).
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

    if path_walker is None:
        pred = sat.predecessor_matrix

        def _walk(i: int, e: int) -> tuple[tuple[int, int], ...] | None:
            return _walk_isl_path(pred, i, e)

        walker = _walk
    else:
        walker = path_walker

    options: list[EgressOption] = []
    for cost, egress, gs_id in candidates[:k]:
        if egress == ingress:
            isl_links: tuple[tuple[int, int], ...] = ()
        else:
            walked = walker(ingress, egress)
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

    Stores the controller's ranked PoP cascade lazily: each entry
    is ``(pop_code, raw_options_tuple, ground_rtt_for_this_pop)``
    where ``raw_options_tuple`` is the cached per-``(ingress, pop)``
    top-K sat list (propagation_rtt = sat segment only, ground_rtt
    = 0 on the raw options). :meth:`RoutingPlaneForward.charge`
    walks the cascade lazily and only materialises a fully-enriched
    :class:`EgressOption` (with uplink + ground baked in) for the
    one option it ultimately chooses.

    This avoids the pre-2026-04-20 pattern where ``decide`` eagerly
    built up to ``max_cascade_pops × k`` enriched options per flow
    (~400 allocations at production scale), most of which the
    first-fit loop never touched.

    ``options`` materialises the full enriched cascade on demand for
    tests and legacy callers; hot-path code iterates via
    :meth:`iter_options` instead.
    """

    user_sat: int
    uplink_rtt: float
    pop_cascade: tuple[tuple[str, tuple[EgressOption, ...], float], ...]

    def __post_init__(self) -> None:
        if not self.pop_cascade or not any(ro for _, ro, _ in self.pop_cascade):
            raise ValueError(
                "PathDecision.pop_cascade must contain at least one "
                "(pop, raw_options, ground_rtt) entry with non-empty "
                "raw_options; decide() should return None rather than "
                "constructing an empty cascade",
            )

    def iter_options(self) -> Iterator[EgressOption]:
        """Yield each option with uplink + ground RTT baked in."""
        for pop_code, raw_opts, ground_rtt in self.pop_cascade:
            for raw in raw_opts:
                yield EgressOption(
                    pop_code=raw.pop_code,
                    egress_sat=raw.egress_sat,
                    gs_id=raw.gs_id,
                    isl_links=raw.isl_links,
                    propagation_rtt=self.uplink_rtt + raw.propagation_rtt,
                    ground_rtt=ground_rtt,
                )

    @property
    def options(self) -> tuple[EgressOption, ...]:
        return tuple(self.iter_options())


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
    """Controller-committed PoP cascade with per-PoP multi-egress reroute.

    Per-flow phases:

    1. ``decide`` — cell→ranked PoP cascade via
       ``CellToPopTable.pops_of(cell, dest)``. Builds a *lazy*
       :class:`PathDecision` holding (per PoP) the cached raw
       options tuple plus that PoP's ground RTT; no per-option
       :class:`EgressOption` is allocated yet.

    2. ``charge`` — walk the cascade (PoPs in rank order, sats in
       E2E order within each PoP); pick the first whose
       ``egress_sat`` has remaining sat-feeder capacity (20 Gbps
       per Ka antenna) and allocate the enriched EgressOption right
       then. If every option's egress sat is saturated, pick the
       option with the smallest current load ratio and accept the
       overflow (still only one allocation).

    3. ``measure`` — use the chosen option's path metrics + the
       final :class:`UsageBook` state to compute per-link
       queuing/loss/bottleneck.

    The ``(ingress, pop) → tuple[EgressOption, ...]`` cache is
    populated lazily on first use within a ``RoutingPlaneForward``
    instance and naturally bounded — the engine constructs a fresh
    instance per epoch, so no cross-epoch leakage.
    """

    __slots__ = (
        "_book", "_decision_cache", "_gf_perf_cache", "_grid",
        "_ground_rtt_cache", "_isl_perf_cache", "_k",
        "_max_cascade_pops", "_options_cache", "_path_cache", "_plane",
        "_sf_perf_cache",
    )

    def __init__(
        self,
        routing_plane: RoutingPlane,
        cell_grid: CellGrid,
        usage_book: UsageBook,
        path_table: object | None = None,    # ignored — kept for callers
        k: int = 8,
        max_cascade_pops: int | None = None,
    ) -> None:
        del path_table  # legacy positional arg from the pre-multi-egress API
        self._plane = routing_plane
        self._grid = cell_grid
        self._book = usage_book
        self._k = k
        # Cap how many ranked PoPs decide() considers per flow.
        # ``None`` (default) means "the full controller cascade" —
        # cell's geographic fallback chain goes all the way until
        # capacity is found. The lazy cascade representation
        # (pop_cascade = tuple of (pop, raw_options, ground_rtt))
        # means the decide-time cost is proportional to the number
        # of PoPs in the cascade, not options; the per-option
        # EgressOption allocation happens only in charge() for the
        # single chosen option, so full-cascade planning is
        # affordable.
        self._max_cascade_pops = (
            max_cascade_pops
            if max_cascade_pops is not None
            else 1 << 30
        )
        self._options_cache: dict[tuple[int, str], tuple[EgressOption, ...]] = {}
        # Decisions depend only on (ingress_sat, cell_id, dest) — uplink
        # RTT is a property of the ingress (cached per src by realize),
        # cascade PoPs come from the plane, options per (ingress, pop)
        # are already memoised in ``_options_cache``, and ground RTTs
        # per (pop, dest) are deterministic at the snapshot level. So
        # every flow that shares (ingress, cell, dest) — which at scale
        # is most of them because subendpoints of the same city share
        # cell + ingress — can reuse the same ``PathDecision`` instead
        # of rebuilding up to 400 EgressOption objects per flow.
        self._decision_cache: dict[
            tuple[int, int, str], PathDecision | None
        ] = {}
        # Sample each (pop, dest) ground RTT once per epoch instead
        # of per flow. LogNormal sampling in GeographicGroundDelay
        # was the dominant cost: 30 k flows × 50 PoPs ≈ 1.5 M RNG
        # draws = ~1 s. Per-flow ground-RTT jitter isn't load-bearing
        # for any downstream metric at the epoch aggregation level.
        self._ground_rtt_cache: dict[tuple[str, str], float] = {}
        # Per-realize ISL path cache. _walk_isl_path depends only on
        # ``(ingress, egress)`` for a fixed snapshot, but the same
        # ``(ingress, egress)`` pair is visited by every PoP whose
        # attached GSs include that egress sat — without this cache
        # we see ~125 k walks for ~10 k unique pairs per realize
        # (≈ 1.8 s of pure Python iteration).
        self._path_cache: dict[
            tuple[int, int], tuple[tuple[int, int], ...] | None
        ] = {}
        # measure() pass-2 caches. Book is frozen across pass 2 so
        # per-link performance depends only on the link id. ~45 k
        # link_performance calls collapse to ~1 k unique links.
        self._isl_perf_cache: dict[tuple[int, int], LinkPerformance] = {}
        self._sf_perf_cache: dict[int, LinkPerformance] = {}
        self._gf_perf_cache: dict[str, LinkPerformance] = {}

    def _options_for(
        self, ingress: int, pop_code: str, snapshot: NetworkSnapshot,
    ) -> tuple[EgressOption, ...]:
        key = (ingress, pop_code)
        cached = self._options_cache.get(key)
        if cached is None:
            pred = snapshot.satellite.predecessor_matrix
            path_cache = self._path_cache

            def walker(i: int, e: int) -> tuple[tuple[int, int], ...] | None:
                wk = (i, e)
                if wk in path_cache:
                    return path_cache[wk]
                result = _walk_isl_path(pred, i, e)
                path_cache[wk] = result
                return result

            cached = compute_egress_options(
                snapshot, ingress, pop_code, self._k, path_walker=walker,
            )
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

        # ── cell → controller-chosen ranked PoP cascade ──
        try:
            cell_id = self._grid.cell_of(flow_key.src)
        except KeyError:
            return None

        cache_key = (ingress, cell_id, flow_key.dst)
        cached = self._decision_cache.get(cache_key)
        if cached is not None or cache_key in self._decision_cache:
            return cached

        try:
            pop_codes = self._plane.cell_to_pop.pops_of(
                cell_id, dest=flow_key.dst,
            )
        except KeyError:
            self._decision_cache[cache_key] = None
            return None
        if not pop_codes:
            self._decision_cache[cache_key] = None
            return None

        ground_truth = context.ground_knowledge.estimator
        if ground_truth is None:
            return None

        uplink_rtt = uplink.delay * 2

        # Build the cascade lazily: each entry is
        # ``(pop_code, raw_options_for_this_pop, ground_rtt)``. The
        # raw options come from the per-``(ingress, pop)`` cache
        # (shared across every flow originating at this ingress),
        # so no per-flow allocation happens here beyond the cascade
        # tuple itself. ``charge`` materialises a single enriched
        # :class:`EgressOption` only for the option it picks.
        #
        # PoPs whose ground-delay estimator raises or whose egress
        # options are empty are skipped — the cascade just gets
        # shorter for this flow rather than rejecting it.
        cascade: list[tuple[str, tuple[EgressOption, ...], float]] = []
        dst = flow_key.dst
        rtt_cache = self._ground_rtt_cache
        options_cache = self._options_cache
        path_cache = self._path_cache
        pred = snapshot.satellite.predecessor_matrix
        k = self._k
        # Cascade loop runs up to 48 × 12 k flows = ~600 k
        # iterations per epoch, so the lookups are inlined to drop
        # one Python frame per iteration; the options/path caches
        # are accessed directly instead of through ``_options_for``.
        for pop_code in pop_codes[: self._max_cascade_pops]:
            key = (pop_code, dst)
            ground_rtt = rtt_cache.get(key)
            if ground_rtt is None:
                try:
                    ground_rtt = ground_truth.estimate(pop_code, dst) * 2
                except KeyError:
                    _log.debug(
                        "decide: no ground estimate for (%s, %s); "
                        "skipping in cell-%s cascade",
                        pop_code, dst, cell_id,
                    )
                    continue
                rtt_cache[key] = ground_rtt
            opts_key = (ingress, pop_code)
            raw_opts = options_cache.get(opts_key)
            if raw_opts is None:
                def walker(
                    i: int, e: int,
                    _pc: dict[tuple[int, int],
                              tuple[tuple[int, int], ...] | None] = path_cache,
                    _pred=pred,
                ) -> tuple[tuple[int, int], ...] | None:
                    wk = (i, e)
                    if wk in _pc:
                        return _pc[wk]
                    result = _walk_isl_path(_pred, i, e)
                    _pc[wk] = result
                    return result

                raw_opts = compute_egress_options(
                    snapshot, ingress, pop_code, k, path_walker=walker,
                )
                options_cache[opts_key] = raw_opts
            if not raw_opts:
                continue
            cascade.append((pop_code, raw_opts, ground_rtt))

        if not cascade:
            self._decision_cache[cache_key] = None
            return None
        decision = PathDecision(
            user_sat=ingress,
            uplink_rtt=uplink_rtt,
            pop_cascade=tuple(cascade),
        )
        self._decision_cache[cache_key] = decision
        return decision

    def charge(
        self, decision: PathDecision, flow_demand: float,
    ) -> EgressOption:
        book = self._book
        view = book.view
        used = book.sat_feeder_used
        uplink_rtt = decision.uplink_rtt

        # Per-sat Ka feeder cap; constant 20 Gbps for the default
        # shell but read via CapacityView for forward-compat.
        for pop_code, raw_opts, ground_rtt in decision.pop_cascade:
            for raw in raw_opts:
                sat = raw.egress_sat
                if used.get(sat, 0.0) + flow_demand <= view.sat_feeder_cap(sat):
                    chosen = EgressOption(
                        pop_code=raw.pop_code, egress_sat=sat,
                        gs_id=raw.gs_id, isl_links=raw.isl_links,
                        propagation_rtt=uplink_rtt + raw.propagation_rtt,
                        ground_rtt=ground_rtt,
                    )
                    self._do_charge(chosen, flow_demand)
                    return chosen

        # Every option's egress sat is saturated. Pick the option
        # with the smallest post-charge load ratio so overflow
        # spreads evenly instead of piling on one sat. For the
        # default uniform 20 Gbps cap this is the same as
        # minimising raw load; generalises to "smallest ratio" if
        # shells ever advertise per-sat caps.
        best_raw: EgressOption | None = None
        best_pop = ""
        best_ground = 0.0
        best_ratio = float("inf")
        for pop_code, raw_opts, ground_rtt in decision.pop_cascade:
            for raw in raw_opts:
                sat = raw.egress_sat
                ratio = used.get(sat, 0.0) / max(view.sat_feeder_cap(sat), 1e-9)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_raw = raw
                    best_pop = pop_code
                    best_ground = ground_rtt

        assert best_raw is not None  # pop_cascade invariant: non-empty
        chosen = EgressOption(
            pop_code=best_pop, egress_sat=best_raw.egress_sat,
            gs_id=best_raw.gs_id, isl_links=best_raw.isl_links,
            propagation_rtt=uplink_rtt + best_raw.propagation_rtt,
            ground_rtt=best_ground,
        )
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
        book = self._book
        view = book.view
        isl_cache = self._isl_perf_cache
        sf_cache = self._sf_perf_cache
        gf_cache = self._gf_perf_cache

        hop_losses: list[float] = []
        hop_capacities: list[float] = []
        total_queuing_oneway = 0.0
        total_tx_oneway = 0.0

        # The usage book is frozen across pass 2 so per-link performance
        # depends only on the link id. Caching here collapses ~45 k
        # link_performance calls (mostly ISL hops revisited by many
        # flows) down to ~1 k unique links per realize.
        for a, b in chosen.isl_links:
            key = (a, b)
            perf = isl_cache.get(key)
            if perf is None:
                isl_cap = view.isl_cap(a, b)
                isl_load = book.isl_used.get(book.isl_key(a, b), 0.0)
                perf = link_performance(
                    float(sat.delay_matrix[a, b]), isl_cap, isl_load,
                )
                isl_cache[key] = perf
            total_queuing_oneway += perf.queuing_ms
            total_tx_oneway += perf.transmission_ms
            hop_losses.append(perf.loss_probability)
            hop_capacities.append(view.isl_cap(a, b))

        egress_sat = chosen.egress_sat
        sf_perf = sf_cache.get(egress_sat)
        if sf_perf is None:
            sf_cap = view.sat_feeder_cap(egress_sat)
            sf_load = book.sat_feeder_used.get(egress_sat, 0.0)
            sf_perf = link_performance(0.0, sf_cap, sf_load)
            sf_cache[egress_sat] = sf_perf
        total_queuing_oneway += sf_perf.queuing_ms
        total_tx_oneway += sf_perf.transmission_ms
        hop_losses.append(sf_perf.loss_probability)
        hop_capacities.append(view.sat_feeder_cap(egress_sat))

        gs_id = chosen.gs_id
        gf_perf = gf_cache.get(gs_id)
        if gf_perf is None:
            gf_cap = view.gs_feeder_cap(gs_id)
            gf_load = book.gs_feeder_used.get(gs_id, 0.0)
            gf_perf = link_performance(0.0, gf_cap, gf_load)
            gf_cache[gs_id] = gf_perf
        total_queuing_oneway += gf_perf.queuing_ms
        total_tx_oneway += gf_perf.transmission_ms
        hop_losses.append(gf_perf.loss_probability)
        hop_capacities.append(view.gs_feeder_cap(gs_id))

        queuing_rtt = total_queuing_oneway * 2
        transmission_rtt = total_tx_oneway * 2
        satellite_rtt = chosen.propagation_rtt + queuing_rtt + transmission_rtt

        return ResolvedFlow(
            pop_code=chosen.pop_code,
            gs_id=gs_id,
            user_sat=decision.user_sat,
            egress_sat=egress_sat,
            satellite_rtt=satellite_rtt,
            ground_rtt=chosen.ground_rtt,
            propagation_rtt=chosen.propagation_rtt,
            queuing_rtt=queuing_rtt,
            transmission_rtt=transmission_rtt,
            loss_probability=path_loss(hop_losses),
            bottleneck_gbps=bottleneck_capacity(hop_capacities),
        )
