"""Tests for forward.py data plane."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from vantage.engine.context import RunContext
from vantage.forward import realize
from vantage.domain import (
    AccessLink,
    Endpoint,
    FlowKey,
    GSPoPEdge,
    GatewayAttachments,
    GroundStation,
    ISLEdge,
    ISLGraph,
    InfrastructureView,
    NetworkSnapshot,
    PathAllocation,
    PoP,
    RoutingIntent,
    SatelliteState,
    TrafficDemand,
)
from vantage.world.ground import GroundKnowledge, HaversineDelay


@pytest.fixture
def simple_snapshot() -> NetworkSnapshot:
    """2-sat network with 1 GS, 1 PoP."""
    graph = ISLGraph(
        shell_id=1, timeslot=0, num_sats=2,
        edges=(ISLEdge(0, 1, 1.0, 300.0, "intra_orbit"),),
    )
    positions = np.array([[0.0, 0.0, 550.0], [1.0, 1.0, 550.0]])
    delay_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    pred_matrix = np.array([[0, 0], [1, 1]], dtype=np.int32)

    gw = GatewayAttachments(attachments=MappingProxyType({
        "gs1": (AccessLink(sat_id=1, elevation_deg=80.0, slant_range_km=560.0, delay=1.87),),
    }))
    sat = SatelliteState(
        positions=positions, graph=graph,
        delay_matrix=delay_matrix, predecessor_matrix=pred_matrix,
        gateway_attachments=gw,
    )
    gs = GroundStation("gs1", 0.5, 0.5, "XX", "Test", 8, 25.0, True, 2.1, 1.3, 25000.0, 32000.0, False)
    pop = PoP("pop1", "pop", "POP", 0.5, 0.5)
    edge = GSPoPEdge("gs1", "pop", 10.0, 0.05, 100.0)
    infra = InfrastructureView(
        pops=(pop,), ground_stations=(gs,), gs_pop_edges=(edge,),
    )

    return NetworkSnapshot(epoch=0, time_s=0.0, satellite=sat, infra=infra)


class _StubWorld:
    """Stub world with no calibration, for unit tests that call realize()."""
    calibration = None


@pytest.fixture
def simple_context() -> RunContext:
    """Minimal RunContext with endpoints and empty knowledge."""
    endpoints = {
        "user_a": Endpoint("user_a", 0.0, 0.0),
        "google": Endpoint("google", 0.5, 0.5),
    }
    return RunContext(
        world=_StubWorld(),  # type: ignore[arg-type]
        endpoints=endpoints,
        ground_knowledge=GroundKnowledge(),
    )


@pytest.mark.unit
class TestForward:

    def test_basic_flow(
        self, simple_snapshot: NetworkSnapshot, simple_context: RunContext,
    ) -> None:
        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "google"): 0.01,
        }))
        intent = RoutingIntent(epoch=0, allocations=MappingProxyType({
            FlowKey("user_a", "google"): PathAllocation(pop_code="pop", gs_id="gs1", user_sat=0, egress_sat=1),
        }))
        simple_context.ground_knowledge.put("pop", "google", 5.0)

        result = realize(intent, simple_snapshot, demand, simple_context)

        assert len(result.flow_outcomes) == 1
        flow = result.flow_outcomes[0]
        assert flow.pop_code == "pop"
        assert flow.demand_gbps == 0.01
        assert flow.total_rtt > 0
        assert flow.ground_rtt == 5.0

    def test_total_equals_sum(
        self, simple_snapshot: NetworkSnapshot, simple_context: RunContext,
    ) -> None:
        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "google"): 0.01,
        }))
        intent = RoutingIntent(epoch=0, allocations=MappingProxyType({
            FlowKey("user_a", "google"): PathAllocation(pop_code="pop", gs_id="gs1", user_sat=0, egress_sat=1),
        }))
        simple_context.ground_knowledge.put("pop", "google", 5.0)

        result = realize(intent, simple_snapshot, demand, simple_context)
        flow = result.flow_outcomes[0]

        expected = flow.satellite_rtt + flow.ground_rtt
        assert abs(flow.total_rtt - expected) < 1e-9

    def test_unrouted_flow(
        self, simple_snapshot: NetworkSnapshot, simple_context: RunContext,
    ) -> None:
        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "google"): 0.01,
            FlowKey("user_b", "google"): 0.02,  # no endpoint
        }))
        intent = RoutingIntent(epoch=0, allocations=MappingProxyType({
            FlowKey("user_a", "google"): PathAllocation(pop_code="pop", gs_id="gs1", user_sat=0, egress_sat=1),
        }))
        simple_context.ground_knowledge.put("pop", "google", 5.0)

        result = realize(intent, simple_snapshot, demand, simple_context)
        assert abs(result.routed_demand_gbps - 0.01) < 1e-9
        assert abs(result.unrouted_demand_gbps - 0.02) < 1e-9

    def test_no_knowledge_zero_ground(
        self, simple_snapshot: NetworkSnapshot,
    ) -> None:
        """Without estimator, cache miss gives ground_rtt = 0.0."""
        ctx = RunContext(
            world=_StubWorld(),  # type: ignore[arg-type]
            endpoints={"user_a": Endpoint("user_a", 0.0, 0.0), "google": Endpoint("google", 0.5, 0.5)},
            ground_knowledge=GroundKnowledge(),  # no estimator
        )
        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "google"): 0.01,
        }))
        intent = RoutingIntent(epoch=0, allocations=MappingProxyType({
            FlowKey("user_a", "google"): PathAllocation(pop_code="pop", gs_id="gs1", user_sat=0, egress_sat=1),
        }))

        result = realize(intent, simple_snapshot, demand, ctx)
        flow = result.flow_outcomes[0]
        assert flow.ground_rtt == 0.0

    def test_l2_fallback_produces_nonzero_ground(
        self, simple_snapshot: NetworkSnapshot,
    ) -> None:
        """With estimator, cache miss produces nonzero ground_rtt."""
        ctx = RunContext(
            world=_StubWorld(),  # type: ignore[arg-type]
            endpoints={
                "user_a": Endpoint("user_a", 0.0, 0.0),
                "google": Endpoint("google", 37.4, -122.1),
            },
            ground_knowledge=GroundKnowledge(estimator=HaversineDelay()),
        )
        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "google"): 0.01,
        }))
        intent = RoutingIntent(epoch=0, allocations=MappingProxyType({
            FlowKey("user_a", "google"): PathAllocation(pop_code="pop", gs_id="gs1", user_sat=0, egress_sat=1),
        }))

        result = realize(intent, simple_snapshot, demand, ctx)
        flow = result.flow_outcomes[0]
        assert flow.ground_rtt > 0
        assert flow.total_rtt == flow.satellite_rtt + flow.ground_rtt
