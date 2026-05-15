"""Forwarding execution loop."""

from __future__ import annotations

import time
from types import MappingProxyType

from vantage.common import DEFAULT_MIN_ELEVATION_DEG
from vantage.common.link_model import pftk_throughput
from vantage.forward.execution.context import RunContext
from vantage.forward.results.models import EpochResult, FlowOutcome
from vantage.forward.strategy.routing import (
    EgressOption,
    ForwardStrategy,
    NoForwardCapacity,
    PathDecision,
)
from vantage.model.network import NetworkSnapshot
from vantage.model.satellite.state import AccessLink
from vantage.model.satellite.visibility import SphericalAccessModel
from vantage.traffic.types import Endpoint, FlowKey, TrafficDemand

__all__ = ["effective_throughput", "realize"]


def find_ingress_satellite(
    src: Endpoint,
    sat_positions: object,
    *,
    rng: object | None = None,
    _visible: list[AccessLink] | None = None,
) -> AccessLink | None:
    """Deterministic forward ingress picker: highest visible elevation.

    The old runner-level name is kept so existing tests and ad-hoc
    monkeypatches still have a stable hook, but the behavior no longer
    performs the 80/20 stochastic choice.
    """
    del src, sat_positions, rng
    if not _visible:
        return None
    return _visible[0]

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
    :func:`realize` call (cached in ``_uplink_cache``). The choice is
    deterministic: first sight within a routing-plane window picks the
    highest-elevation visible satellite, then the strategy may pin that
    sat for the rest of the window while it remains visible. This keeps
    planned control outputs aligned with forward without reintroducing
    the old 80/20 stochastic branch.
    """
    del ingress_seed_base  # kept for API compatibility with existing callers
    sat = snapshot.satellite
    total_demand = 0.0

    _access = SphericalAccessModel()
    _visible_cache: dict[str, tuple[AccessLink, ...]] = {}
    _uplink_cache: dict[object, AccessLink | None] = {}
    _get_pin = getattr(strategy, "uplink_sat_pin", None)
    _uplink_sat_pin: dict[str, int] | None = (
        _get_pin() if callable(_get_pin) else None
    )
    _preferred_ingress = getattr(strategy, "preferred_ingress", None)
    _uses_preferred_ingress = callable(_preferred_ingress)

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
        uplink_cache_key: object = (
            (src_name, flow_key.dst) if _uses_preferred_ingress else src_name
        )
        if uplink_cache_key not in _uplink_cache:
            t0 = _perf()
            if src_name not in _visible_cache:
                _visible_cache[src_name] = _access.compute_access(
                    src_ep.lat_deg, src_ep.lon_deg, 0.0, sat.positions,
                    DEFAULT_MIN_ELEVATION_DEG,
                )
            visible = _visible_cache[src_name]
            link: AccessLink | None = None
            if visible:
                preferred_sat = (
                    _preferred_ingress(
                        flow_key, src_ep, visible, snapshot, context, demand.epoch,
                    )
                    if _uses_preferred_ingress
                    else None
                )
                if preferred_sat is not None:
                    for candidate in visible:
                        if candidate.sat_id == preferred_sat:
                            link = candidate
                            break
                if link is None:
                    pinned_sat = (
                        None if _uplink_sat_pin is None
                        else _uplink_sat_pin.get(src_name)
                    )
                    if pinned_sat is not None:
                        for candidate in visible:
                            if candidate.sat_id == pinned_sat:
                                link = candidate
                                break
                if link is None:
                    link = find_ingress_satellite(
                        src_ep, sat.positions, _visible=visible,
                    )
                    if (
                        link is not None
                        and _uplink_sat_pin is not None
                        and not _uses_preferred_ingress
                    ):
                        _uplink_sat_pin[src_name] = link.sat_id
            _uplink_cache[uplink_cache_key] = link
            ingress_s += _perf() - t0
        uplink = _uplink_cache[uplink_cache_key]
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
        try:
            chosen = strategy.charge(decision, flow_demand)
        except NoForwardCapacity:
            charge_s += _perf() - t0
            continue
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
