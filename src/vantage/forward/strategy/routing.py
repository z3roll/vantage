"""Forwarding strategy: controller plane to per-flow path decisions."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from vantage.common.link_model import LinkPerformance
from vantage.control.plane import RoutingPlane
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

__all__ = [
    "EgressOption",
    "ForwardStrategy",
    "PathDecision",
    "RoutingPlaneForward",
]

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
    return tuple(zip(rev_path, rev_path[1:], strict=False))

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
        "_sat_paths", "_sf_perf_cache", "_template_cache",
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
    #   * ``_options_by_ingress`` — top-K enriched options per
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
        if previous is not None and previous._plane is routing_plane:
            previous.reset_for_epoch(usage_book)
            return previous
        return cls(
            routing_plane, cell_grid, usage_book,
            k=k, max_cascade_pops=max_cascade_pops,
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

        * ``_options_by_ingress`` — top-K options per ``(ingress, pop)``
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
        self._decision_cache.clear()
        self._ground_rtt_by_dst.clear()
        self._isl_perf_cache.clear()
        self._sf_perf_cache.clear()
        self._gf_perf_cache.clear()

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
        book = self._book
        view = book.view
        used = book.sat_feeder_used
        uplink_rtt = decision.uplink_rtt

        # Per-sat Ka feeder cap; constant 20 Gbps for the default
        # shell but read via CapacityView for forward-compat.
        for _pop_code, raw_opts, ground_rtt in decision.pop_cascade:
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
