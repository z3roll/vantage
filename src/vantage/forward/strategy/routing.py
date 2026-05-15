"""Forwarding strategy: controller plane to per-flow path decisions."""

from __future__ import annotations

import logging
import heapq
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from vantage.common.link_model import LinkPerformance
from vantage.control.plane import PlannedPath, RoutingPlane
from vantage.forward.execution.measurement import measure_flow
from vantage.forward.resources.accounting import UsageBook
from vantage.forward.results.models import ResolvedFlow
from vantage.model.coverage import CellGrid
from vantage.model.network import NetworkSnapshot
from vantage.model.satellite.state import AccessLink
from vantage.traffic.types import Endpoint, FlowKey

if TYPE_CHECKING:
    from vantage.forward.execution.context import RunContext

_log = logging.getLogger(__name__)

_CAP_EPS = 1e-9

__all__ = [
    "EgressOption",
    "ForwardStrategy",
    "NoForwardCapacity",
    "PathDecision",
    "PlannedRoutingPlaneForward",
    "RoutingPlaneForward",
]


class NoForwardCapacity(RuntimeError):
    """Raised when a flow has no capacity-feasible forward path."""

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
    many of these per flow: ``NamedTuple.__new__`` is ~3× faster than
    the frozen-dataclass constructor at our scale, and the tuple backing lets
    ``min(options, key=...)`` iterate without attribute lookups.
    """

    pop_code: str
    egress_sat: int
    gs_id: str
    isl_links: tuple[tuple[int, int], ...]
    propagation_rtt: float    # uplink + sat-segment (RTT)
    ground_rtt: float         # PoP→destination (RTT)


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
    # Store hops in the UsageBook's canonical ISL key form. Forward's
    # capacity loop treats ISLs as undirected resources, so this avoids
    # re-normalising every hop on every candidate check.
    return tuple(
        (a, b) if a <= b else (b, a)
        for a, b in zip(rev_path, rev_path[1:], strict=False)
    )


def _shortest_capacity_isl_path(
    adjacency: Mapping[int, tuple[tuple[int, float], ...]],
    book: UsageBook,
    src: int,
    dst: int,
    demand_gbps: float,
) -> tuple[float, tuple[tuple[int, int], ...]] | None:
    """Shortest currently capacity-feasible ISL path.

    This is the scalable fallback when the controller's shortest path
    is blocked: remove ISL edges with insufficient residual capacity
    and run one Dijkstra over the frozen adjacency view.
    """
    if src == dst:
        return (0.0, ())

    seq = 0
    heap: list[tuple[float, int, int]] = [(0.0, seq, src)]
    dist: dict[int, float] = {src: 0.0}
    prev: dict[int, int] = {}

    while heap:
        delay, _seq, node = heapq.heappop(heap)
        if delay != dist.get(node):
            continue
        if node == dst:
            rev_path = [dst]
            cur = dst
            while cur != src:
                cur = prev[cur]
                rev_path.append(cur)
            rev_path.reverse()
            return delay, tuple(
                (a, b) if a <= b else (b, a)
                for a, b in zip(rev_path, rev_path[1:], strict=False)
            )

        for neighbor, edge_delay in adjacency.get(node, ()):
            edge_key = (
                (node, neighbor) if node <= neighbor else (neighbor, node)
            )
            is_saturated = demand_gbps > 0.0 and edge_key in book.saturated_isl
            if is_saturated or book.remaining_isl_key(edge_key) < demand_gbps:
                continue
            next_delay = delay + edge_delay
            old_delay = dist.get(neighbor)
            if old_delay is not None and next_delay >= old_delay:
                continue
            seq += 1
            dist[neighbor] = next_delay
            prev[neighbor] = node
            heapq.heappush(heap, (next_delay, seq, neighbor))
    return None

@dataclass(frozen=True, slots=True)
class PathDecision:
    """Path-level routing decision for a flow.

    Stores the controller's ranked PoP cascade lazily: each entry
    is ``(pop_code, raw_options_tuple, ground_rtt_for_this_pop)``
    where ``raw_options_tuple`` is the cached per-``(ingress, pop)``
    ranked egress list (propagation_rtt = sat segment only, ground_rtt
    = 0 on the raw options). :meth:`RoutingPlaneForward.charge`
    walks the cascade lazily and only materialises a fully-enriched
    :class:`EgressOption` (with uplink + ground baked in) for the
    one option it ultimately chooses.

    This avoids the pre-2026-04-20 pattern where ``decide`` eagerly
    built hundreds of enriched options per flow, most of which the
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
        for _pop_code, raw_opts, ground_rtt in self.pop_cascade:
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
        """Pick a capacity-feasible option from ``decision``.

        The chosen option must have both sat-feeder and ISL headroom.
        If none exists, implementations raise :class:`NoForwardCapacity`
        and the flow remains unrouted.

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

class RoutingPlaneForward:
    """Controller-committed PoP cascade with per-PoP multi-egress reroute.

    Per-flow phases:

    1. ``decide`` — cell→ranked PoP cascade via
       ``CellToPopTable.pops_of(cell, dest)``. Builds a *lazy*
       :class:`PathDecision` holding (per PoP) the cached raw
       options tuple plus that PoP's ground RTT; no per-option
       :class:`EgressOption` is allocated yet.

    2. ``charge`` — walk the cascade (PoPs in rank order, sats in
       E2E order within each PoP); pick the first whose ISL path and
       ``egress_sat`` have remaining capacity. If no candidate has
       both sat-feeder and ISL headroom, the flow is left unrouted.

    3. ``measure`` — use the chosen option's path metrics + the
       final :class:`UsageBook` state to compute per-link
       queuing/loss/bottleneck.

    The ``(ingress, pop) → tuple[EgressOption, ...]`` cache is
    populated lazily on first use within a ``RoutingPlaneForward``
    instance and naturally bounded — the engine constructs a fresh
    instance per epoch, so no cross-epoch leakage.
    """

    __slots__ = (
        "_all_saturated_raw_opts", "_book", "_capacity_path_cache",
        "_capacity_path_fail_cache", "_decision_cache", "_enforce_isl_capacity",
        "_forward_counters", "_gf_perf_cache", "_grid", "_ground_rtt_by_dst",
        "_isl_perf_cache", "_max_cascade_pops", "_options_by_ingress",
        "_path_cache", "_plane", "_pop_egress", "_pred_row_cache",
        "_sat_feeder_fail_threshold_by_raw_opts", "_sat_paths", "_sf_perf_cache",
        "_template_cache",
        "_uplink_sat_pin",
    )

    def __init__(
        self,
        routing_plane: RoutingPlane,
        cell_grid: CellGrid,
        usage_book: UsageBook,
        path_table: object | None = None,    # ignored — kept for callers
        k: int = 8,
        max_cascade_pops: int | None = None,
        enforce_isl_capacity: bool = True,
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
        self._enforce_isl_capacity = enforce_isl_capacity
        del k  # legacy constructor knob; per-GS top-k is owned by GatewayAttachments.
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
        self._capacity_path_cache: dict[
            tuple[int, int], tuple[float, tuple[tuple[int, int], ...]]
        ] = {}
        self._capacity_path_fail_cache: dict[tuple[int, int], float] = {}
        self._all_saturated_raw_opts: set[int] = set()
        self._sat_feeder_fail_threshold_by_raw_opts: dict[int, float] = {}
        self._forward_counters: dict[str, int] = {}
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
        # Per-(ingress, cell, dst) "route template" — the plane-derived
        # skeleton of a decision: the ordered list of
        # ``(pop_code, raw_opts)`` pairs that survived the empty-egress
        # filter at first decide(). ``None`` is an explicit "no route"
        # marker so we don't re-probe pops_of() every epoch for dead
        # pairs.
        #
        # Survives :meth:`reset_for_epoch` because its inputs are all
        # plane-bound: the ranked cascade comes from
        # ``_plane.cell_to_pop.pops_of`` (frozen for the refresh
        # window), and ``raw_opts`` comes from the plane-static
        # ``_options_by_ingress``. What the template intentionally
        # does *not* carry is the per-epoch stuff — uplink RTT and
        # GK-derived ground RTT are stitched in fresh each epoch so
        # the feedback-learned GK state still moves decisions and the
        # measurement pipeline still sees current-epoch truth.
        self._template_cache: dict[
            tuple[int, int, str],
            tuple[tuple[str, tuple[EgressOption, ...]], ...] | None,
        ] = {}
        # Per-``src`` pinned ingress sat id, scoped to the refresh
        # window. :func:`realize` reads/writes this in place so a
        # terminal's ingress stays fixed across every cached-plan
        # epoch in the window — the stochastic 80/20 branch of
        # :func:`find_ingress_satellite` fires only on first sight
        # (or when the pinned sat drops below horizon). Survives
        # :meth:`reset_for_epoch` (window-static); drops automatically
        # when ``for_epoch`` rebuilds the forward on plane refresh,
        # which is exactly the point where a new window begins.
        #
        # Window key: the forward instance's lifetime (= plane
        # identity via :meth:`for_epoch`). NOT tied to the REFRESH
        # constant and NOT an injected parameter — it piggybacks on
        # the existing plane-refresh invalidation point so run.py
        # stays thin.
        self._uplink_sat_pin: dict[str, int] = {}

    # ── Cache-reuse API: hoist RoutingPlaneForward out of the per-epoch
    # loop and reuse the *plane-static* routing caches across
    # cached-plan epochs. Only caches whose inputs depend purely on
    # the bound plane survive a :meth:`reset_for_epoch` call:
    #
    #   * ``_options_by_ingress`` — ranked egress options per
    #     ``(ingress, pop)``; inputs are ``pop_egress`` + the
    #     ``delay_row``/``pred_row`` from ``sat_paths``, both owned
    #     by the plane.
    #   * ``_path_cache`` — ISL hop sequences per ``(ingress, egress)``;
    #     driven by the plane's ``pred_row``.
    #   * ``_pred_row_cache`` — memoised ``.tolist()`` rows from the
    #     plane's ``sat_paths.pred_row``.
    #   * ``_template_cache`` — the per-``(ingress, cell, dst)`` route
    #     template (ordered ``(pop, raw_opts)`` list), derived solely
    #     from ``_plane.cell_to_pop`` + ``_options_by_ingress``. Reused
    #     across every cached-plan epoch within a refresh window.
    #   * ``_uplink_sat_pin`` — per-``src`` pinned ingress sat id for
    #     the window. :func:`realize` reuses the pinned sat if still
    #     visible; falls back to :func:`find_ingress_satellite` (and
    #     re-pins) only when the pin drops below horizon.
    #
    # Caches that bake *per-epoch* inputs must be cleared every epoch,
    # even when the plane is reused:
    #
    #   * ``_decision_cache`` — each :class:`PathDecision` stores
    #     ``uplink_rtt = uplink.delay * 2`` taken from the epoch's
    #     :class:`AccessLink` (sat positions move between epochs, so
    #     ``delay`` differs even when the ingress sat id is the same)
    #     plus a cascade whose per-PoP ``ground_rtt`` was resolved
    #     from the current :class:`GroundKnowledge` state (which the
    #     feedback loop mutates every epoch).
    #   * ``_ground_rtt_by_dst`` — memoises
    #     :meth:`GroundKnowledge.get_or_estimate` results, which
    #     evolve across epochs as learned stats accumulate.
    #   * ``_isl_perf_cache`` / ``_sf_perf_cache`` / ``_gf_perf_cache``
    #     — derived from the previous epoch's :class:`UsageBook` state.
    #   * ``_capacity_path_cache`` — residual-capacity alternate ISL
    #     paths; valid only while the current epoch's book is bound.
    #
    # Callers MUST re-construct a new :class:`RoutingPlaneForward`
    # when the plane changes (identity check via :attr:`plane` or
    # version check via :attr:`plane_version`).

    @property
    def plane(self) -> RoutingPlane:
        """The :class:`RoutingPlane` this forward is bound to."""
        return self._plane

    def uplink_sat_pin(self) -> dict[str, int]:
        """Window-scoped ``src_name → pinned_sat_id`` map shared with
        :func:`realize`. Caller reads/writes in place; the dict's
        lifetime is the forward instance (i.e. the refresh window)."""
        return self._uplink_sat_pin

    @property
    def plane_version(self) -> int:
        """Version tag of the bound plane. Callers that want to
        decide whether to reuse vs. rebuild the forward should
        compare ``bl_forward.plane is bl_plane`` (identity) or
        ``bl_forward.plane_version == bl_plane.version`` (value)."""
        return int(self._plane.version)

    @classmethod
    def for_epoch(
        cls,
        previous: RoutingPlaneForward | None,
        routing_plane: RoutingPlane,
        cell_grid: CellGrid,
        usage_book: UsageBook,
        *,
        k: int = 8,
        max_cascade_pops: int | None = None,
        enforce_isl_capacity: bool = True,
    ) -> RoutingPlaneForward:
        """Return a forward wired to ``usage_book`` for this epoch.

        Centralises the cache-reuse policy so callers (``run.py`` /
        the engine) never have to reason about plane identity or
        cache lifetimes themselves — they just hand in the previous
        forward (or ``None``) and the plane + book for the epoch
        they're about to realize.

        * If ``previous`` is still bound to ``routing_plane``
          (identity check), the plane-static caches carry over and
          only the per-epoch ones are dropped via
          :meth:`reset_for_epoch`.
        * Otherwise a fresh instance is constructed with empty
          caches.
        """
        if (
            previous is not None
            and previous._plane is routing_plane
            and previous._enforce_isl_capacity == enforce_isl_capacity
        ):
            previous.reset_for_epoch(usage_book)
            return previous
        return cls(
            routing_plane, cell_grid, usage_book,
            k=k, max_cascade_pops=max_cascade_pops,
            enforce_isl_capacity=enforce_isl_capacity,
        )

    def reset_for_epoch(self, usage_book: UsageBook) -> None:
        """Re-bind the per-epoch :class:`UsageBook` and drop every
        cache whose contents depend on per-epoch inputs.

        Cleared:

        * ``_decision_cache`` — each :class:`PathDecision` bakes the
          current epoch's ``uplink_rtt`` and GK-derived per-PoP
          ``ground_rtt`` into its cascade, so a cached decision from
          epoch *t* is not a safe decision at epoch *t+1*.
        * ``_ground_rtt_by_dst`` — memoised GK scores evolve as the
          feedback loop mutates :class:`GroundKnowledge`.
        * ``_isl_perf_cache`` / ``_sf_perf_cache`` / ``_gf_perf_cache``
          — derived from the previous epoch's :class:`UsageBook`.

        Preserved (plane-static):

        * ``_options_by_ingress`` — ranked options per ``(ingress, pop)``
          come from the plane's ``pop_egress`` + ``sat_paths``.
        * ``_path_cache`` — ISL hop sequences via the plane's
          ``pred_row``.
        * ``_pred_row_cache`` — memoised rows of the plane's
          ``sat_paths.pred_row``.
        * ``_template_cache`` — per-``(ingress, cell, dst)`` route
          template reused across every cached-plan epoch within a
          refresh window (see the class-level comment for contents).
        * ``_uplink_sat_pin`` — per-``src`` ingress sat id pinned for
          the refresh window; keeps the ingress deterministic across
          cached-plan epochs (see the class-level comment).
        """
        self._book = usage_book
        self._all_saturated_raw_opts.clear()
        self._sat_feeder_fail_threshold_by_raw_opts.clear()
        self._capacity_path_cache.clear()
        self._capacity_path_fail_cache.clear()
        self._decision_cache.clear()
        self._forward_counters.clear()
        self._ground_rtt_by_dst.clear()
        self._isl_perf_cache.clear()
        self._sf_perf_cache.clear()
        self._gf_perf_cache.clear()

    def forward_counters(self) -> Mapping[str, int]:
        """Per-epoch debug counters collected by the charge hot path."""
        return self._forward_counters

    def _count(self, name: str, amount: int = 1) -> None:
        counters = self._forward_counters
        counters[name] = counters.get(name, 0) + amount

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
        """Ranked egress options for ``(ingress, pop_code)``.

        Pulls the per-PoP downlink candidates from the controller-built
        :class:`PopEgressTable`, then vectorises the per-candidate
        ISL RTT against the per-ingress row from :class:`SatPathTable`
        and sorts every finite survivor by E2E cost. The per-GS
        candidate fanout is already capped by ``GatewayAttachments``
        (currently top-8 visible sats per GS), so this method must not
        apply another global per-PoP top-k that would hide later GSs.
        The snapshot parameter is kept
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
        order_local = np.argsort(valid_cost, kind="stable")
        top_idx = valid_idx[order_local]
        options: list[EgressOption] = []
        path_cache = self._path_cache
        pred_row = self._pred_row_for(ingress)
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

        # ── Plane-static template: built once, reused across every
        # cached-plan epoch within the refresh window. Contains only
        # the plane-derived skeleton — the ordered cascade of PoPs
        # that have non-empty raw_opts, paired with the raw options
        # themselves. Uplink RTT and per-PoP ground RTT stay out of
        # the template so the feedback-learned GK and per-epoch
        # geometry can still drive each epoch's PathDecision.
        template = self._build_or_get_template(
            ingress, cell_id, flow_key.dst, snapshot,
        )
        if template is None:
            self._decision_cache[cache_key] = None
            return None

        # ── Per-epoch stitch: uplink RTT (from the epoch's AccessLink)
        # + per-PoP ground RTT (from the epoch's GroundKnowledge).
        # ``_ground_rtt_by_dst`` is cleared every reset_for_epoch so
        # these values track the current GK state.
        dst = flow_key.dst
        ground_knowledge = context.ground_knowledge
        uplink_rtt = uplink.delay * 2
        rtt_by_pop = self._ground_rtt_by_dst.get(dst)
        if rtt_by_pop is None:
            rtt_by_pop = {}
            self._ground_rtt_by_dst[dst] = rtt_by_pop

        cascade: list[tuple[str, tuple[EgressOption, ...], float]] = []
        for pop_code, raw_opts in template:
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

    def _build_or_get_template(
        self,
        ingress: int,
        cell_id: int,
        dst: str,
        snapshot: NetworkSnapshot,
    ) -> tuple[tuple[str, tuple[EgressOption, ...]], ...] | None:
        """Return the plane-static route template for
        ``(ingress, cell_id, dst)`` — ordered ``(pop, raw_opts)`` pairs
        with empty-egress PoPs already filtered out. Cached across
        epochs because every input (plane cell_to_pop, plane-derived
        raw_opts) is frozen for the refresh window; per-epoch state
        (uplink, GK) is deliberately absent so the caller can stitch
        those in fresh each epoch.

        ``None`` is an explicit "no route" marker — an empty-cascade
        pair stays in the cache so we don't re-probe
        :meth:`CellToPopTable.pops_of` every epoch for dead pairs.
        """
        key = (ingress, cell_id, dst)
        template = self._template_cache.get(key)
        if template is not None or key in self._template_cache:
            return template

        try:
            pop_codes = self._plane.cell_to_pop.pops_of(cell_id, dest=dst)
        except KeyError:
            self._template_cache[key] = None
            return None
        if not pop_codes:
            self._template_cache[key] = None
            return None

        opts_by_pop = self._options_by_ingress.get(ingress)
        if opts_by_pop is None:
            opts_by_pop = {}
            self._options_by_ingress[ingress] = opts_by_pop

        entries: list[tuple[str, tuple[EgressOption, ...]]] = []
        for pop_code in pop_codes[: self._max_cascade_pops]:
            raw_opts = opts_by_pop.get(pop_code)
            if raw_opts is None:
                raw_opts = self._options_for(
                    ingress, pop_code, snapshot, opts_by_pop=opts_by_pop,
                )
            if not raw_opts:
                continue
            entries.append((pop_code, raw_opts))

        template = tuple(entries) if entries else None
        self._template_cache[key] = template
        return template

    def charge(
        self, decision: PathDecision, flow_demand: float,
    ) -> EgressOption:
        self._count("charge_calls")
        uplink_rtt = decision.uplink_rtt
        ingress = decision.user_sat

        for _pop_code, raw_opts, ground_rtt in decision.pop_cascade:
            self._count("pop_scans")
            chosen = self._choose_in_pop(
                raw_opts, ingress, uplink_rtt, ground_rtt, flow_demand,
                require_sat_feeder=True,
            )
            if chosen is not None:
                self._do_charge(chosen, flow_demand)
                return chosen

        _log.debug(
            "charge: no sat-feeder/ISL-capacity-feasible option for %.6f Gbps flow",
            flow_demand,
        )
        self._count("no_capacity")
        raise NoForwardCapacity(
            "no sat-feeder/ISL-capacity-feasible option for flow demand "
            f"{flow_demand:.6f} Gbps"
        )

    def _choose_in_pop(
        self,
        raw_opts: tuple[EgressOption, ...],
        ingress: int,
        uplink_rtt: float,
        ground_rtt: float,
        flow_demand: float,
        *,
        require_sat_feeder: bool,
    ) -> EgressOption | None:
        """Return the lowest-RTT capacity-feasible option in one PoP.

        ``raw_opts`` contains the shortest ISL path per downlink
        candidate. A min-heap lazily inserts one residual-feasible
        alternate for a candidate only when its controller-shortest
        path is blocked by ISL capacity, preserving global order across
        ``GS × egress × path`` without eagerly materialising alternates.
        """
        raw_opts_key = id(raw_opts)
        if (
            require_sat_feeder
            and flow_demand > 0.0
            and raw_opts_key in self._all_saturated_raw_opts
        ):
            self._count("all_saturated_pop_skips")
            return None
        failed_at = self._sat_feeder_fail_threshold_by_raw_opts.get(raw_opts_key)
        if failed_at is not None and flow_demand >= failed_at - _CAP_EPS:
            self._count("sat_feeder_fail_cache_hits")
            return None

        alternate_heap: list[tuple[float, int, EgressOption]] = []
        next_primary = 0
        raw_count = len(raw_opts)
        all_primary_sats_saturated = require_sat_feeder and flow_demand > 0.0
        any_sat_feeder_ok = not require_sat_feeder or flow_demand <= 0.0

        while next_primary < raw_count or alternate_heap:
            if next_primary < raw_count:
                primary = raw_opts[next_primary]
                primary_key = (primary.propagation_rtt, next_primary)
            else:
                primary = None
                primary_key = (float("inf"), 1 << 62)

            if alternate_heap and (alternate_heap[0][0], alternate_heap[0][1]) < primary_key:
                _cost, seq, candidate = heapq.heappop(alternate_heap)
                is_alternate = True
            else:
                assert primary is not None
                seq = next_primary
                candidate = primary
                next_primary += 1
                is_alternate = False

            self._count("options_scanned")
            sat = candidate.egress_sat
            if require_sat_feeder:
                if (
                    flow_demand > 0.0
                    and sat in self._book.saturated_sat_feeders
                ):
                    self._count("sat_feeder_saturated_skips")
                    continue
                all_primary_sats_saturated = False
                if not self._book.can_charge_sat_feeder(sat, flow_demand):
                    self._count("sat_feeder_rejects")
                    continue
                any_sat_feeder_ok = True
            if not self._enforce_isl_capacity:
                self._count("isl_capacity_disabled_accepts")
                return EgressOption(
                    pop_code=candidate.pop_code,
                    egress_sat=sat,
                    gs_id=candidate.gs_id,
                    isl_links=candidate.isl_links,
                    propagation_rtt=uplink_rtt + candidate.propagation_rtt,
                    ground_rtt=ground_rtt,
                )
            self._count("isl_path_checks")
            if self._book.can_charge_isl_path_keys(
                candidate.isl_links, flow_demand,
            ):
                return EgressOption(
                    pop_code=candidate.pop_code,
                    egress_sat=sat,
                    gs_id=candidate.gs_id,
                    isl_links=candidate.isl_links,
                    propagation_rtt=uplink_rtt + candidate.propagation_rtt,
                    ground_rtt=ground_rtt,
                )
            self._count("isl_rejects")
            if not is_alternate:
                self._count("alternate_requests")
                next_variant = self._capacity_feasible_variant(
                    candidate, ingress, flow_demand,
                )
                if next_variant is not None:
                    heapq.heappush(
                        alternate_heap,
                        (next_variant.propagation_rtt, seq, next_variant),
                    )
        if all_primary_sats_saturated:
            self._all_saturated_raw_opts.add(raw_opts_key)
        if require_sat_feeder and flow_demand > 0.0 and not any_sat_feeder_ok:
            old_failed_at = self._sat_feeder_fail_threshold_by_raw_opts.get(raw_opts_key)
            self._sat_feeder_fail_threshold_by_raw_opts[raw_opts_key] = (
                flow_demand
                if old_failed_at is None
                else min(old_failed_at, flow_demand)
            )
        return None

    def _first_isl_feasible_variant(
        self, raw: EgressOption, ingress: int, flow_demand: float,
    ) -> EgressOption | None:
        if not self._enforce_isl_capacity:
            return raw
        if self._book.can_charge_isl_path_keys(raw.isl_links, flow_demand):
            return raw
        return self._capacity_feasible_variant(raw, ingress, flow_demand)

    def _capacity_feasible_variant(
        self, raw: EgressOption, ingress: int, flow_demand: float,
    ) -> EgressOption | None:
        adjacency = self._sat_paths.isl_adjacency
        if not adjacency:
            return None
        key = (ingress, raw.egress_sat)
        failed_at = self._capacity_path_fail_cache.get(key)
        if failed_at is not None and flow_demand >= failed_at - _CAP_EPS:
            self._count("alternate_fail_cache_hits")
            return None
        found = self._capacity_path_cache.get(key)
        from_cache = False
        if found is not None:
            if self._book.can_charge_isl_path_keys(found[1], flow_demand):
                from_cache = True
                self._count("alternate_success_cache_hits")
            else:
                found = None
        if found is None:
            self._count("alternate_dijkstra_calls")
            found = _shortest_capacity_isl_path(
                adjacency, self._book, ingress, raw.egress_sat, flow_demand,
            )
            if found is not None:
                self._capacity_path_cache[key] = found
        if found is None:
            old_failed_at = self._capacity_path_fail_cache.get(key)
            self._capacity_path_fail_cache[key] = (
                flow_demand
                if old_failed_at is None
                else min(old_failed_at, flow_demand)
            )
            self._count("alternate_fail")
            return None
        if not from_cache:
            self._count("alternate_success")
        isl_delay, isl_links = found
        primary_delay = self._sat_paths.isl_delay(ingress, raw.egress_sat)
        return raw._replace(
            isl_links=isl_links,
            propagation_rtt=float(
                raw.propagation_rtt + (isl_delay - primary_delay) * 2.0
            ),
        )

    def _do_charge(self, option: EgressOption, demand: float) -> None:
        if self._enforce_isl_capacity:
            for key in option.isl_links:
                self._book.charge_isl_key(key, demand)
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
        # Per-link perf caches are owned by the forward (tied to its
        # UsageBook lifecycle via :meth:`reset_for_epoch`); the actual
        # measurement lives in :mod:`vantage.forward.execution.measurement`.
        return measure_flow(
            decision, chosen,
            book=self._book,
            sat_paths=self._sat_paths,
            isl_cache=self._isl_perf_cache,
            sf_cache=self._sf_perf_cache,
            gf_cache=self._gf_perf_cache,
            ground_rtt_truth=ground_rtt_truth,
        )


class PlannedRoutingPlaneForward:
    """Execute controller-planned concrete paths without capacity search.

    This is the companion forwarder for path-aware optimizers. The
    optimizer has already selected ``(PoP, GS, egress_sat, ISL path)``
    candidates under its capacity model, so the data plane should not
    run another first-fit capacity search that changes the plan. It
    simply resolves the planned path for ``(ingress, cell, dest)``,
    charges the UsageBook for measurement, and reuses
    :func:`measure_flow` for final RTT/loss/bottleneck statistics.

    Existing PoP-cascade policies continue to use
    :class:`RoutingPlaneForward`.
    """

    __slots__ = (
        "_book", "_decision_cache", "_forward_counters", "_gf_perf_cache",
        "_grid", "_ground_rtt_by_dst", "_isl_perf_cache", "_plane",
        "_preferred_ingress_by_cell_dest", "_sf_perf_cache", "_sat_paths",
        "_uplink_sat_pin",
    )

    def __init__(
        self,
        routing_plane: RoutingPlane,
        cell_grid: CellGrid,
        usage_book: UsageBook,
    ) -> None:
        if routing_plane.path_hints is None:
            raise ValueError(
                "PlannedRoutingPlaneForward requires RoutingPlane.path_hints"
            )
        self._plane = routing_plane
        self._sat_paths = routing_plane.sat_paths
        self._grid = cell_grid
        self._book = usage_book
        self._decision_cache: dict[
            tuple[int, int, str], PathDecision | None
        ] = {}
        self._ground_rtt_by_dst: dict[str, dict[str, float]] = {}
        self._isl_perf_cache: dict[tuple[int, int], LinkPerformance] = {}
        self._sf_perf_cache: dict[int, LinkPerformance] = {}
        self._gf_perf_cache: dict[str, LinkPerformance] = {}
        self._forward_counters: dict[str, int] = {}
        self._uplink_sat_pin: dict[str, int] = {}
        hints = routing_plane.path_hints
        assert hints is not None
        preferred: dict[tuple[int, str], int] = {}
        for ingress, cell_id, dest in hints.paths:
            preferred.setdefault((cell_id, dest), ingress)
        self._preferred_ingress_by_cell_dest = preferred

    @classmethod
    def for_epoch(
        cls,
        previous: PlannedRoutingPlaneForward | None,
        routing_plane: RoutingPlane,
        cell_grid: CellGrid,
        usage_book: UsageBook,
    ) -> PlannedRoutingPlaneForward:
        """Reuse the planned forward while the same plane is active."""
        if previous is not None and previous._plane is routing_plane:
            previous.reset_for_epoch(usage_book)
            return previous
        return cls(routing_plane, cell_grid, usage_book)

    @property
    def plane(self) -> RoutingPlane:
        return self._plane

    @property
    def plane_version(self) -> int:
        return int(self._plane.version)

    def uplink_sat_pin(self) -> dict[str, int]:
        """Window-scoped ``src_name -> pinned_sat_id`` map."""
        return self._uplink_sat_pin

    def preferred_ingress(
        self,
        flow_key: FlowKey,
        src_ep: Endpoint,
        visible: tuple[AccessLink, ...] | list[AccessLink],
        snapshot: NetworkSnapshot,
        context: RunContext,
        epoch: int,
    ) -> int | None:
        """Return the controller-planned ingress for this flow."""
        del src_ep, snapshot, context, epoch
        try:
            cell_id = self._grid.cell_of(flow_key.src)
        except KeyError:
            return None
        preferred = self._preferred_ingress_by_cell_dest.get((cell_id, flow_key.dst))
        if preferred is None:
            return None
        visible_ids = {link.sat_id for link in visible}
        return preferred if preferred in visible_ids else None

    def reset_for_epoch(self, usage_book: UsageBook) -> None:
        self._book = usage_book
        self._decision_cache.clear()
        self._ground_rtt_by_dst.clear()
        self._isl_perf_cache.clear()
        self._sf_perf_cache.clear()
        self._gf_perf_cache.clear()
        self._forward_counters.clear()

    def forward_counters(self) -> Mapping[str, int]:
        """Per-epoch debug counters collected by the planned hot path."""
        return self._forward_counters

    def _count(self, name: str, amount: int = 1) -> None:
        counters = self._forward_counters
        counters[name] = counters.get(name, 0) + amount

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
        del src_ep, snapshot, epoch
        try:
            cell_id = self._grid.cell_of(flow_key.src)
        except KeyError:
            self._count("missing_cell")
            return None

        key = (ingress, cell_id, flow_key.dst)
        cached = self._decision_cache.get(key)
        if cached is not None or key in self._decision_cache:
            return cached

        hints = self._plane.path_hints
        assert hints is not None
        planned_paths = hints.paths_for(ingress, cell_id, flow_key.dst)
        if not planned_paths:
            self._count("missing_plan")
            self._decision_cache[key] = None
            return None

        dst = flow_key.dst
        rtt_by_pop = self._ground_rtt_by_dst.get(dst)
        if rtt_by_pop is None:
            rtt_by_pop = {}
            self._ground_rtt_by_dst[dst] = rtt_by_pop
        ground_knowledge = context.ground_knowledge

        cascade: list[tuple[str, tuple[EgressOption, ...], float]] = []
        for planned in planned_paths:
            raw = self._raw_option_from_plan(planned)
            if raw is None:
                continue
            ground_rtt = rtt_by_pop.get(planned.pop_code)
            if ground_rtt is None:
                try:
                    ground_rtt = ground_knowledge.get_or_estimate(
                        planned.pop_code, dst,
                    )
                except KeyError:
                    self._count("missing_ground")
                    continue
                rtt_by_pop[planned.pop_code] = ground_rtt
            cascade.append((planned.pop_code, (raw,), ground_rtt))

        if not cascade:
            self._count("invalid_plan")
            self._decision_cache[key] = None
            return None
        decision = PathDecision(
            user_sat=ingress,
            uplink_rtt=uplink.delay * 2,
            pop_cascade=tuple(cascade),
        )
        self._decision_cache[key] = decision
        return decision

    def _raw_option_from_plan(self, planned: PlannedPath) -> EgressOption | None:
        access_rtt = planned.access_rtt
        if access_rtt is None:
            access_rtt = self._lookup_access_rtt(planned)
            if access_rtt is None:
                self._count("missing_access")
                return None

        isl_rtt = 0.0
        for a, b in planned.isl_links:
            one_way = self._direct_isl_delay(a, b)
            if one_way is None:
                self._count("invalid_isl_link")
                return None
            isl_rtt += one_way * 2.0

        return EgressOption(
            pop_code=planned.pop_code,
            egress_sat=planned.egress_sat,
            gs_id=planned.gs_id,
            isl_links=planned.isl_links,
            propagation_rtt=isl_rtt + access_rtt,
            ground_rtt=0.0,
        )

    def _lookup_access_rtt(self, planned: PlannedPath) -> float | None:
        egress_ids, base_cost, gs_ids = self._plane.pop_egress.for_pop(
            planned.pop_code,
        )
        for idx, gs_id in enumerate(gs_ids):
            if gs_id == planned.gs_id and int(egress_ids[idx]) == planned.egress_sat:
                return float(base_cost[idx])
        return None

    def _direct_isl_delay(self, sat_a: int, sat_b: int) -> float | None:
        for neighbor, delay in self._sat_paths.neighbors(sat_a):
            if neighbor == sat_b:
                return delay
        return None

    def charge(
        self, decision: PathDecision, flow_demand: float,
    ) -> EgressOption:
        self._count("charge_calls")
        for _pop_code, raw_opts, ground_rtt in decision.pop_cascade:
            if not raw_opts:
                continue
            raw = raw_opts[0]
            chosen = EgressOption(
                pop_code=raw.pop_code,
                egress_sat=raw.egress_sat,
                gs_id=raw.gs_id,
                isl_links=raw.isl_links,
                propagation_rtt=decision.uplink_rtt + raw.propagation_rtt,
                ground_rtt=ground_rtt,
            )
            self._do_charge(chosen, flow_demand)
            return chosen
        self._count("no_planned_path")
        raise NoForwardCapacity("no planned path for flow")

    def _do_charge(self, option: EgressOption, demand: float) -> None:
        for key in option.isl_links:
            self._book.charge_isl_key(key, demand)
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
        del snapshot
        return measure_flow(
            decision, chosen,
            book=self._book,
            sat_paths=self._sat_paths,
            isl_cache=self._isl_perf_cache,
            sf_cache=self._sf_perf_cache,
            gf_cache=self._gf_perf_cache,
            ground_rtt_truth=ground_rtt_truth,
        )
