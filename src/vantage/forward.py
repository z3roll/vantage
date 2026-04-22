"""Data plane: flow-level PoP selection + delay computation.

**RoutingPlaneForward**: the controller pre-commits a per-(cell, dest)
ranked PoP cascade plus two sat-level routing artifacts
(:class:`SatPathTable` for ISL RTTs / predecessors and
:class:`PopEgressTable` for per-PoP downlink candidates). The data
plane reads those artifacts through the routing plane — it no longer
touches ``snapshot.satellite.delay_matrix`` /
``predecessor_matrix`` / ``gateway_attachments`` directly on the hot
path. What it still owns at runtime: ingress resolution from user
visibility, per-flow load-aware option picking, and the per-hop
queuing/loss measurement against the current :class:`UsageBook`.

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
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from vantage.common import DEFAULT_MIN_ELEVATION_DEG
from vantage.common.seed import mix_seed
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

    Controller-side helper. After the routing-plane refactor the data
    plane consumes :class:`PopEgressTable` + :class:`SatPathTable` from
    the plane and no longer calls this function; it remains here as
    the planner's egress-enumeration primitive (used by
    :class:`ProgressiveController`'s per-(cell, pop) capacity check).

    Enumerates every ``(gs, sat)`` pair such that ``gs`` is attached
    to ``pop_code`` and ``sat`` is a visible egress for ``gs``,
    computes the round-trip cost ``ISL + downlink + backhaul``, sorts
    ascending, takes the top *K*. Each returned option's
    ``propagation_rtt`` carries only the *sat segment* — the caller
    adds the uplink RTT before reporting the user-facing latency.
    ``ground_rtt`` is set to 0 here; the caller fills it from the
    ground-delay estimator (which depends on the destination, not
    the route).

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

    ``pred[s, t]`` is "predecessor of t on the shortest path from s".
    This wrapper materialises ``pred[src]`` as a Python list (native
    integer access is ~2× faster than ``int(numpy_scalar)``) and then
    defers to :func:`_walk_isl_path_row`. Hot-path callers that walk
    many destinations from the same ``src`` should cache the row and
    call the row variant directly to pay the ``.tolist()`` cost only
    once per ingress."""
    if src == dst:
        return ()
    return _walk_isl_path_row(pred[src].tolist(), src, dst)


def _walk_isl_path_row(
    pred_row: list[int], src: int, dst: int,
) -> tuple[tuple[int, int], ...] | None:
    """Walk ``pred_row[cur]`` back from ``dst`` until we hit ``src``.

    ``pred_row`` is expected to be ``pred[src].tolist()`` — a flat
    Python list of predecessor ids. Using native list indexing drops
    the per-hop numpy scalar cost that dominated the old walker at
    scale (82 k walks × 30 hops ≈ 2.5 M numpy reads/epoch)."""
    if src == dst:
        return ()
    rev_path: list[int] = [dst]
    cur = dst
    n = len(pred_row)
    for _ in range(n):
        prev = pred_row[cur]
        if prev < 0:
            return None
        rev_path.append(prev)
        if prev == src:
            break
        cur = prev
    else:
        return None
    rev_path.reverse()
    # ``zip(rev_path, rev_path[1:])`` is the C-implemented equivalent
    # of the earlier ``((p[i], p[i+1]) for i in range(...))`` genexpr
    # and trims ~40 % off per-hop overhead at 1.6 M iterations/epoch.
    return tuple(zip(rev_path, rev_path[1:]))


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

    # Intentionally no ``__post_init__`` validation: the generator
    # walk over ``pop_cascade`` cost ~130 ms/epoch at production scale
    # for a check the builder (``decide``) already satisfies — it
    # returns ``None`` before reaching this constructor whenever the
    # cascade would be empty. Frozen+slots already locks the shape.

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
        snapshot: NetworkSnapshot, *, ground_rtt_truth: float | None,
    ) -> ResolvedFlow:
        """Compute per-link queuing/loss/bottleneck for ``chosen``
        using the *current* :class:`UsageBook` state.

        ``ground_rtt_truth`` is the :class:`GroundTruth` sample for
        the chosen ``(pop_code, dst, epoch)``. Implementations use it
        as the realized ground RTT instead of the decide-time prior;
        when ``None`` (no truth source configured, e.g. in unit
        tests), fall back to ``chosen.ground_rtt``.

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
    *,
    ingress_seed_base: int = 0,
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

    The stochastic branch uses a per-realize RNG seeded from
    ``(ingress_seed_base, demand.epoch)`` via
    :func:`vantage.common.seed.mix_seed`. Two controllers called with
    the same ``ingress_seed_base`` for the same epoch draw the same
    ingress-sat assignments, while changing ``ingress_seed_base``
    across runs (derived from the run-level seed) varies the
    stochastic ingress selection between runs.
    """
    sat = snapshot.satellite
    total_demand = 0.0

    _access = SphericalAccessModel()
    _visible_cache: dict[str, list[AccessLink]] = {}
    _uplink_cache: dict[str, AccessLink | None] = {}
    _ingress_rng = _random.Random(mix_seed(ingress_seed_base, demand.epoch))

    pending: list[tuple[FlowKey, float, PathDecision, EgressOption]] = []

    # Cumulative phase timers. ``perf_counter`` is ~50 ns per call; at
    # ~10⁵ flows/epoch the per-phase bracketing overhead is <10 ms,
    # well under the per-realize budget. Local aliases keep the inner
    # loop free of attribute lookups.
    _perf = time.perf_counter
    t_total_start = _perf()
    ingress_s = 0.0
    decide_s = 0.0
    charge_s = 0.0
    measure_s = 0.0

    # ── Pass 1: decide + charge ──
    for flow_key, flow_demand in demand.flows.items():
        total_demand += flow_demand

        src_ep = context.endpoints.get(flow_key.src)
        if src_ep is None:
            continue

        src_name = flow_key.src
        if src_name not in _uplink_cache:
            t0 = _perf()
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
            ingress_s += _perf() - t0
        uplink = _uplink_cache[src_name]
        if uplink is None:
            continue

        t0 = _perf()
        decision = strategy.decide(
            flow_key, src_ep, uplink.sat_id, uplink,
            snapshot, context, demand.epoch,
        )
        decide_s += _perf() - t0
        if decision is None:
            continue

        t0 = _perf()
        chosen = strategy.charge(decision, flow_demand)
        charge_s += _perf() - t0
        pending.append((flow_key, flow_demand, decision, chosen))

    # ── Pass 2: measure with final loads + emit outcomes ──
    # ``ground_truth`` (when present on the context) is sampled per
    # chosen (pop, dst) at the current epoch to produce the realized
    # ground RTT that the data plane reports. When absent (test
    # fixtures without a truth source) measure falls back to the
    # decide-time ground RTT. The planner's prior/stats are NEVER
    # read here — the data plane's output is truth, and feedback is
    # what ties truth back into the planner's knowledge.
    truth = getattr(context, "ground_truth", None)
    outcomes: list[FlowOutcome] = []
    routed_demand = 0.0
    for flow_key, flow_demand, decision, chosen in pending:
        t0 = _perf()
        truth_rtt: float | None
        if truth is not None:
            try:
                # ``flow_key.src`` is the flow identity axis — two
                # flows with the same ``(pop, dst)`` but different
                # sources get independent draws; the same flow
                # reproduces across runs with the same ``seed_base``.
                truth_rtt = truth.sample(
                    chosen.pop_code, flow_key.dst, demand.epoch, flow_key.src,
                )
            except KeyError:
                truth_rtt = None
        else:
            truth_rtt = None
        resolved = strategy.measure(
            decision, chosen, snapshot, ground_rtt_truth=truth_rtt,
        )
        measure_s += _perf() - t0
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

    total_s = _perf() - t_total_start
    forward_timing = MappingProxyType({
        "total_ms": total_s * 1000.0,
        "ingress_ms": ingress_s * 1000.0,
        "decide_ms": decide_s * 1000.0,
        "charge_ms": charge_s * 1000.0,
        "measure_ms": measure_s * 1000.0,
    })

    return EpochResult(
        epoch=demand.epoch,
        flow_outcomes=tuple(outcomes),
        total_demand_gbps=total_demand,
        routed_demand_gbps=routed_demand,
        unrouted_demand_gbps=total_demand - routed_demand,
        forward_timing_ms=forward_timing,
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
        "_ground_rtt_by_dst", "_isl_perf_cache",
        "_k", "_max_cascade_pops", "_options_by_ingress",
        "_path_cache", "_plane", "_pop_egress", "_pred_row_cache",
        "_sat_paths", "_sf_perf_cache",
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
        # Hold direct references to the two controller-built routing
        # artifacts the data plane consumes. The hot path reads ISL
        # RTTs + hop reconstruction from ``_sat_paths`` and per-PoP
        # downlink candidate tables from ``_pop_egress``; it never
        # reaches back into ``snapshot.satellite`` on the forwarding
        # path. Stored on the instance so inner loops don't pay an
        # extra ``self._plane.X`` traversal per call.
        self._sat_paths = routing_plane.sat_paths
        self._pop_egress = routing_plane.pop_egress
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
        # Decisions depend only on (ingress_sat, cell_id, dest) — uplink
        # RTT is a property of the ingress (cached per src by realize),
        # cascade PoPs come from the plane, options per (ingress, pop)
        # are memoised in ``_options_by_ingress``, and ground RTTs per
        # (pop, dest) in ``_ground_rtt_by_dst``. Every flow sharing
        # (ingress, cell, dest) — which at scale is most of them
        # because subendpoints of the same city share cell + ingress —
        # reuses the same ``PathDecision`` instead of rebuilding up to
        # 400 EgressOption objects per flow.
        self._decision_cache: dict[
            tuple[int, int, str], PathDecision | None
        ] = {}
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
        # Per-realize predecessor-row cache: controller hands us the
        # numpy row via ``sat_paths.pred_row``; we memoise its
        # ``.tolist()`` form because native Python list indexing is
        # ~2× faster per hop than numpy-scalar access at 2.5 M
        # hops/epoch.
        self._pred_row_cache: dict[int, list[int]] = {}
        # Two-level caches keyed on the loop-invariant (``ingress`` /
        # ``dst``). decide()'s 48-PoP cascade runs ~1.4 M times per
        # epoch; looking up via a plain ``str`` pop key against a
        # per-ingress or per-dst sub-dict avoids the tuple allocation
        # that a ``dict[(str, str)]`` lookup would incur per iteration.
        self._options_by_ingress: dict[
            int, dict[str, tuple[EgressOption, ...]]
        ] = {}
        self._ground_rtt_by_dst: dict[str, dict[str, float]] = {}

    def _pred_row_for(self, ingress: int) -> list[int]:
        """Return ``pred[ingress]`` as a Python list, cached per realize.

        Materialising the row once and sharing it across every walker
        call for the same ingress amortises the ``.tolist()`` cost
        (~100 µs for a 4 k-wide row) against 10–100 walks per
        ingress; the alternative — reading ``pred[src, cur]`` as a
        numpy scalar on every hop — is ~2× slower per access and
        dominates at 2.5 M hops/epoch.

        The source row comes from :attr:`_sat_paths`, the controller's
        sat-path artifact, so the data plane is never touching
        ``snapshot.satellite.predecessor_matrix`` directly."""
        row = self._pred_row_cache.get(ingress)
        if row is None:
            row = self._sat_paths.pred_row(ingress).tolist()
            self._pred_row_cache[ingress] = row
        return row

    def _options_for(
        self,
        ingress: int,
        pop_code: str,
        snapshot: NetworkSnapshot,
        *,
        opts_by_pop: dict[str, tuple[EgressOption, ...]] | None = None,
    ) -> tuple[EgressOption, ...]:
        """Top-K enriched egress options for ``(ingress, pop_code)``.

        Pulls the per-PoP downlink candidates from the controller-built
        :class:`PopEgressTable`, then vectorises the per-candidate
        ISL RTT against the per-ingress row from :class:`SatPathTable`
        and takes the top-K survivors. The snapshot parameter is kept
        only so the ``ForwardStrategy.decide`` signature remains
        compatible for test subclasses; this method itself no longer
        touches ``snapshot.satellite``.

        ``opts_by_pop`` lets decide() pass the already-resolved
        per-ingress sub-dict so we skip the outer
        ``_options_by_ingress[ingress]`` lookup per call; external
        callers can omit it and we resolve it here."""
        del snapshot  # routing inputs all come from the plane now
        if opts_by_pop is None:
            opts_by_pop = self._options_by_ingress.get(ingress)
            if opts_by_pop is None:
                opts_by_pop = {}
                self._options_by_ingress[ingress] = opts_by_pop
        cached = opts_by_pop.get(pop_code)
        if cached is not None:
            return cached
        egress_ids, base_cost, gs_ids = self._pop_egress.for_pop(pop_code)
        if egress_ids.size == 0:
            opts_by_pop[pop_code] = ()
            return ()
        delay_row = self._sat_paths.delay_row(ingress)
        # One-way ISL from ingress to every candidate egress. Numpy
        # fancy-indexes in one pass rather than 950 k scalar reads.
        cost = delay_row[egress_ids] * 2.0 + base_cost
        finite = np.isfinite(cost)
        if not finite.any():
            opts_by_pop[pop_code] = ()
            return ()
        valid_idx = np.nonzero(finite)[0]
        valid_cost = cost[valid_idx]
        k = self._k
        # ``argpartition`` gets the unordered top-K in O(m); then a
        # tiny ``argsort`` over only K entries finishes the ranking.
        # At ``k ≤ m`` this is ~3× faster than sorting the full array.
        if valid_idx.size > k:
            part = np.argpartition(valid_cost, k - 1)[:k]
            order_local = part[np.argsort(valid_cost[part])]
        else:
            order_local = np.argsort(valid_cost)
        top_idx = valid_idx[order_local]
        path_cache = self._path_cache
        pred_row = self._pred_row_for(ingress)
        options: list[EgressOption] = []
        for pos in top_idx:
            i = int(pos)
            egress = int(egress_ids[i])
            gs_id = gs_ids[i]
            if egress == ingress:
                isl_links: tuple[tuple[int, int], ...] = ()
            else:
                wk = (ingress, egress)
                walked = path_cache.get(wk)
                if walked is None and wk not in path_cache:
                    walked = _walk_isl_path_row(pred_row, ingress, egress)
                    path_cache[wk] = walked
                if walked is None:
                    continue
                isl_links = walked
            options.append(EgressOption(
                pop_code=pop_code,
                egress_sat=egress,
                gs_id=gs_id,
                isl_links=isl_links,
                propagation_rtt=float(cost[i]),
                ground_rtt=0.0,
            ))
        result = tuple(options)
        opts_by_pop[pop_code] = result
        return result

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

        ground_knowledge = context.ground_knowledge

        uplink_rtt = uplink.delay * 2

        # Build the cascade lazily: each entry is
        # ``(pop_code, raw_options_for_this_pop, ground_rtt)``. The
        # raw options come from the per-``(ingress, pop)`` cache
        # (shared across every flow originating at this ingress),
        # so no per-flow allocation happens here beyond the cascade
        # tuple itself. ``charge`` materialises a single enriched
        # :class:`EgressOption` only for the option it picks.
        #
        # PoPs whose ground-knowledge service cannot provide an RTT
        # (cache miss + no estimator) or whose egress options are
        # empty are skipped — the cascade just gets shorter for this
        # flow rather than rejecting it.
        cascade: list[tuple[str, tuple[EgressOption, ...], float]] = []
        dst = flow_key.dst
        # Resolve per-ingress / per-dst sub-dicts once per decide call
        # so the 48-PoP cascade loop only touches a single
        # ``dict.get(str)`` per cache — no tuple allocation per
        # iteration. At 1.4 M cascade iterations per epoch the tuple
        # construction was the largest remaining per-iter cost after
        # the earlier pred-row / vectorised-options changes.
        rtt_by_pop = self._ground_rtt_by_dst.get(dst)
        if rtt_by_pop is None:
            rtt_by_pop = {}
            self._ground_rtt_by_dst[dst] = rtt_by_pop
        opts_by_pop = self._options_by_ingress.get(ingress)
        if opts_by_pop is None:
            opts_by_pop = {}
            self._options_by_ingress[ingress] = opts_by_pop
        for pop_code in pop_codes[: self._max_cascade_pops]:
            ground_rtt = rtt_by_pop.get(pop_code)
            if ground_rtt is None:
                try:
                    ground_rtt = ground_knowledge.get_or_estimate(pop_code, dst)
                except KeyError:
                    _log.debug(
                        "decide: no ground RTT in knowledge for (%s, %s); "
                        "skipping in cell-%s cascade",
                        pop_code, dst, cell_id,
                    )
                    continue
                rtt_by_pop[pop_code] = ground_rtt
            raw_opts = opts_by_pop.get(pop_code)
            if raw_opts is None:
                raw_opts = self._options_for(
                    ingress, pop_code, snapshot, opts_by_pop=opts_by_pop,
                )
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
        *,
        ground_rtt_truth: float | None = None,
    ) -> ResolvedFlow:
        del snapshot  # per-hop propagation comes from the plane's sat-path table
        sat_paths = self._sat_paths
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
                    sat_paths.isl_delay(a, b), isl_cap, isl_load,
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

        # Ground RTT reported back to the outer realize loop is the
        # truth sample when available, the planner's decide-time
        # ground_rtt otherwise. Feedback consumes the emitted value
        # to update learned stats, so handing it truth here is what
        # closes the "measure-truth, learn-truth-into-knowledge" loop.
        ground_rtt = (
            ground_rtt_truth if ground_rtt_truth is not None else chosen.ground_rtt
        )

        return ResolvedFlow(
            pop_code=chosen.pop_code,
            gs_id=gs_id,
            user_sat=decision.user_sat,
            egress_sat=egress_sat,
            satellite_rtt=satellite_rtt,
            ground_rtt=ground_rtt,
            propagation_rtt=chosen.propagation_rtt,
            queuing_rtt=queuing_rtt,
            transmission_rtt=transmission_rtt,
            loss_probability=path_loss(hop_losses),
            bottleneck_gbps=bottleneck_capacity(hop_capacities),
        )
