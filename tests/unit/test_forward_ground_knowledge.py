from __future__ import annotations

from types import MappingProxyType

import numpy as np

from vantage.domain import (
    AccessLink,
    CapacityView,
    Cell,
    CellGrid,
    CellToPopTable,
    Endpoint,
    FlowKey,
    PopEgressTable,
    RoutingPlane,
    SatPathTable,
    UsageBook,
)
from vantage.engine.context import RunContext
from vantage.forward import EgressOption, RoutingPlaneForward
from vantage.world.ground import GroundKnowledge


class _CountingEstimator:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def estimate(self, pop_code: str, dest_name: str) -> float:
        del pop_code, dest_name
        self.calls += 1
        return self.value


class _FakeRoutingPlaneForward(RoutingPlaneForward):
    __slots__ = ()

    def _options_for(
        self,
        ingress: int,
        pop_code: str,
        snapshot: object,
        *,
        opts_by_pop: dict[str, tuple[EgressOption, ...]] | None = None,
    ) -> tuple[EgressOption, ...]:
        del snapshot, opts_by_pop
        return (
            EgressOption(
                pop_code=pop_code,
                egress_sat=ingress,
                gs_id="gs-a",
                isl_links=(),
                propagation_rtt=1.0,
                ground_rtt=0.0,
            ),
        )


def _empty_sat_paths(n_sats: int = 1) -> SatPathTable:
    """Minimal valid :class:`SatPathTable` stub for tests.

    ``_FakeRoutingPlaneForward`` overrides ``_options_for`` so the data
    plane never dereferences these arrays, but ``SatPathTable`` still
    validates that both matrices are square and same-shape, so we
    produce a 1×1 pair. ``delay_matrix`` diagonal is 0 and the
    predecessor diagonal is the identity — the Dijkstra convention.
    """
    delay = np.zeros((n_sats, n_sats), dtype=np.float64)
    pred = np.arange(n_sats, dtype=np.int32).reshape(1, n_sats).repeat(n_sats, axis=0)
    return SatPathTable(
        delay_matrix=delay, predecessor_matrix=pred,
        version=0, built_at=0.0,
    )


def _empty_pop_egress() -> PopEgressTable:
    """Minimal :class:`PopEgressTable` with no candidates.

    ``_FakeRoutingPlaneForward._options_for`` returns fabricated
    options, so the table is never queried in the data path. An
    empty mapping is the simplest valid construction.
    """
    return PopEgressTable(
        candidates=MappingProxyType({}), version=0, built_at=0.0,
    )


def _make_forward() -> RoutingPlaneForward:
    cell_id = 101
    grid = CellGrid(
        cells=MappingProxyType({cell_id: Cell(cell_id=cell_id, lat_deg=0.0, lon_deg=0.0)}),
        endpoint_to_cell=MappingProxyType({"src": cell_id}),
    )
    plane = RoutingPlane(
        cell_to_pop=CellToPopTable(
            mapping=MappingProxyType({cell_id: ("pop-a",)}),
            version=0,
            built_at=0.0,
        ),
        sat_paths=_empty_sat_paths(),
        pop_egress=_empty_pop_egress(),
        version=0,
        built_at=0.0,
    )
    book = UsageBook(
        view=CapacityView(
            isl_cap_index=MappingProxyType({}),
            sat_feeder_gbps=20.0,
            gs_by_id=MappingProxyType({}),
        )
    )
    return _FakeRoutingPlaneForward(plane, grid, book)


def test_decide_uses_cached_ground_knowledge_without_estimator() -> None:
    forward = _make_forward()
    knowledge = GroundKnowledge()
    knowledge.put("pop-a", "dst", 42.0)
    context = RunContext(world=object(), endpoints={}, ground_knowledge=knowledge)

    decision = forward.decide(
        FlowKey(src="src", dst="dst"),
        Endpoint(name="src", lat_deg=0.0, lon_deg=0.0),
        ingress=0,
        uplink=AccessLink(sat_id=0, elevation_deg=45.0, slant_range_km=1000.0, delay=1.0),
        snapshot=object(),
        context=context,
        epoch=0,
    )

    assert decision is not None
    assert len(decision.pop_cascade) == 1
    assert decision.pop_cascade[0][0] == "pop-a"
    assert decision.pop_cascade[0][2] == 42.0


def test_decide_prefers_cached_ground_knowledge_over_estimator_sampling() -> None:
    forward = _make_forward()
    estimator = _CountingEstimator(value=99.0)
    knowledge = GroundKnowledge(estimator=estimator)
    knowledge.put("pop-a", "dst", 24.0)
    context = RunContext(world=object(), endpoints={}, ground_knowledge=knowledge)

    decision = forward.decide(
        FlowKey(src="src", dst="dst"),
        Endpoint(name="src", lat_deg=0.0, lon_deg=0.0),
        ingress=0,
        uplink=AccessLink(sat_id=0, elevation_deg=45.0, slant_range_km=1000.0, delay=1.0),
        snapshot=object(),
        context=context,
        epoch=0,
    )

    assert decision is not None
    assert decision.pop_cascade[0][2] == 24.0
    assert estimator.calls == 0


def test_reset_for_epoch_refreshes_ground_rtt_and_uplink_rtt() -> None:
    """Across cached-plan epochs the forward must re-resolve both the
    ground RTT (GK evolves via feedback) and the uplink RTT (sat
    positions move between epochs) rather than returning the cached
    :class:`PathDecision` from the previous epoch.
    """
    forward = _make_forward()
    knowledge = GroundKnowledge()
    knowledge.put("pop-a", "dst", 10.0)
    context = RunContext(world=object(), endpoints={}, ground_knowledge=knowledge)
    flow = FlowKey(src="src", dst="dst")
    src_ep = Endpoint(name="src", lat_deg=0.0, lon_deg=0.0)

    decision_t0 = forward.decide(
        flow, src_ep, ingress=0,
        uplink=AccessLink(sat_id=0, elevation_deg=45.0, slant_range_km=1000.0, delay=1.0),
        snapshot=object(), context=context, epoch=0,
    )
    assert decision_t0 is not None
    assert decision_t0.uplink_rtt == 2.0  # 1.0 * 2
    assert decision_t0.pop_cascade[0][2] == 10.0

    # Mutate what the next epoch would see: a fresh GK score (feedback
    # updates GroundKnowledge every epoch) and a different uplink
    # delay (sat has moved within the same plane).
    knowledge.put("pop-a", "dst", 25.0)

    # Without reset, the decision_cache returns the stale epoch-0 tuple.
    stale = forward.decide(
        flow, src_ep, ingress=0,
        uplink=AccessLink(sat_id=0, elevation_deg=45.0, slant_range_km=1000.0, delay=3.0),
        snapshot=object(), context=context, epoch=1,
    )
    assert stale is decision_t0  # cache hit: proves the reset is necessary

    # After reset the cache is rebuilt from current inputs.
    book = UsageBook(
        view=CapacityView(
            isl_cap_index=MappingProxyType({}),
            sat_feeder_gbps=20.0,
            gs_by_id=MappingProxyType({}),
        )
    )
    forward.reset_for_epoch(book)
    decision_t1 = forward.decide(
        flow, src_ep, ingress=0,
        uplink=AccessLink(sat_id=0, elevation_deg=45.0, slant_range_km=1000.0, delay=3.0),
        snapshot=object(), context=context, epoch=1,
    )
    assert decision_t1 is not None
    assert decision_t1.uplink_rtt == 6.0              # 3.0 * 2, fresh uplink
    # Fresh GK query: whatever GK returns after the second put() —
    # the point is simply that it's not the epoch-0 cached value.
    fresh_rtt = knowledge.get_or_estimate("pop-a", "dst")
    assert decision_t1.pop_cascade[0][2] == fresh_rtt
    assert decision_t1.pop_cascade[0][2] != decision_t0.pop_cascade[0][2]
