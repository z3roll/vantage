"""Data-plane flow measurement: per-hop queuing / loss / throughput.

Executed in :func:`vantage.forward.realize`'s pass 2, after every
flow has been charged against the :class:`~vantage.forward.UsageBook`.
Given a :class:`~vantage.forward.PathDecision` + the chosen
:class:`~vantage.forward.EgressOption` and the final per-link load,
computes the :class:`ResolvedFlow` realize() turns into a
:class:`~vantage.forward.FlowOutcome`.

This logic used to live on ``RoutingPlaneForward.measure``; it lives
here now because it operates against the **final** book state rather
than participating in the forwarding decision. The
``decide`` / ``charge`` phases are about choosing a path; ``measure``
is about reading what actually happened — a concern of the engine's
epoch pipeline, not of the forwarder.

Three per-link performance caches are owned by the caller and passed
in: the book is frozen across pass 2, so the per-link ``LinkPerformance``
depends only on the link id and is shared across every flow that
traverses that link within a single realize() call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vantage.common.link_model import (
    LinkPerformance,
    bottleneck_capacity,
    link_performance,
    path_loss,
)
from vantage.forward.results.models import ResolvedFlow

if TYPE_CHECKING:
    from vantage.control.plane import SatPathTable
    from vantage.forward.resources.accounting import UsageBook
    from vantage.forward.strategy.routing import EgressOption, PathDecision


__all__ = ["measure_flow"]


def measure_flow(
    decision: PathDecision,
    chosen: EgressOption,
    *,
    book: UsageBook,
    sat_paths: SatPathTable,
    isl_cache: dict[tuple[int, int], LinkPerformance],
    sf_cache: dict[int, LinkPerformance],
    gf_cache: dict[str, LinkPerformance],
    ground_rtt_truth: float | None = None,
) -> ResolvedFlow:
    """Compute per-hop queuing/loss/bottleneck for ``chosen`` using
    the current :class:`~vantage.forward.UsageBook` state.

    The three ``*_cache`` dicts are owned by the caller and reused
    across every flow measured in the same realize() pass 2 —
    because the book is frozen across that pass, per-link
    :class:`LinkPerformance` depends only on link id. At production
    scale this collapses ~45 k ``link_performance`` calls down to
    the ~1 k unique-link count.

    ``ground_rtt_truth`` overrides the decide-time ground RTT when
    provided — this is the hook that closes pass 2's
    "measure-truth, learn-truth-into-knowledge" loop against
    :class:`~vantage.model.ground.GroundTruth`. ``None`` (the
    default, used by tests that don't configure truth) falls back
    to ``chosen.ground_rtt``.
    """
    view = book.view
    hop_losses: list[float] = []
    hop_capacities: list[float] = []
    total_queuing_oneway = 0.0
    total_tx_oneway = 0.0

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
