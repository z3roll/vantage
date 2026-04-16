"""Tests for forward.py data plane (RoutingPlaneForward)."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from vantage.domain import (
    AccessLink,
    CapacityView,
    CellGrid,
    CellToPopTable,
    Endpoint,
    FIBEntry,
    FlowKey,
    GSPoPEdge,
    GatewayAttachments,
    GroundStation,
    ISLEdge,
    ISLGraph,
    InfrastructureView,
    NetworkSnapshot,
    PoP,
    RoutingPlane,
    SatelliteFIB,
    SatelliteState,
    ShellConfig,
    TrafficDemand,
    UsageBook,
)
from vantage.engine.context import RunContext
from vantage.forward import RoutingPlaneForward, precompute_path_table, realize
from vantage.world.ground import GroundKnowledge, MeasuredGroundDelay


def _stub_ground_truth() -> MeasuredGroundDelay:
    """Hand-injected measurement table for the simple 2-sat fixture."""
    return MeasuredGroundDelay(
        one_way_rtt_ms={("pop", "google"): 2.5},  # 5 ms round-trip
    )


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
    gs = GroundStation("gs1", 0.5, 0.5, "XX", "Test", 8, 25.0, True, 2.1, 1.3, 80.0, False)
    pop = PoP("pop1", "pop", "POP", 0.5, 0.5)
    edge = GSPoPEdge("gs1", "pop", 10.0, 0.05, 100.0)
    infra = InfrastructureView(
        pops=(pop,), ground_stations=(gs,), gs_pop_edges=(edge,),
    )
    return NetworkSnapshot(epoch=0, time_s=0.0, satellite=sat, infra=infra)


def _build_routing_plane(simple_snapshot: NetworkSnapshot) -> RoutingPlane:
    """Build a minimal RoutingPlane for the 2-sat fixture.

    sat 0 → forward to sat 1 (next_hop=1)
    sat 1 → egress to gs1
    cell maps to pop.
    """
    # FIB: sat 0 forwards to sat 1; sat 1 is egress via gs1
    fib_0 = SatelliteFIB(
        sat_id=0,
        fib=MappingProxyType({
            "pop": FIBEntry.forward(next_hop_sat=1, cost_ms=5.0),
        }),
        version=0, built_at=0.0,
    )
    fib_1 = SatelliteFIB(
        sat_id=1,
        fib=MappingProxyType({
            "pop": FIBEntry.egress(gs_id="gs1", cost_ms=3.74),
        }),
        version=0, built_at=0.0,
    )
    # Cell grid: single cell mapping user_a → pop
    cell_grid = CellGrid.from_endpoints([("user_a", 0.0, 0.0)])
    cell_id = cell_grid.cell_of("user_a")
    cell_to_pop = CellToPopTable(
        mapping=MappingProxyType({cell_id: "pop"}),
        version=0, built_at=0.0,
    )
    return RoutingPlane(
        cell_to_pop=cell_to_pop,
        sat_fibs=MappingProxyType({0: fib_0, 1: fib_1}),
        version=0, built_at=0.0,
    ), cell_grid


class _StubWorld:
    calibration = None
    shell = ShellConfig(
        shell_id=1, num_orbits=1, sats_per_orbit=2,
        altitude_km=550.0, inclination_deg=53.0,
        orbit_cycle_s=6000.0, phase_shift=1,
        feeder_capacity_gbps=80.0,
    )
    ground_stations = ()


@pytest.fixture
def simple_context() -> RunContext:
    endpoints = {
        "user_a": Endpoint("user_a", 0.0, 0.0),
        "google": Endpoint("google", 37.4, -122.1),
    }
    return RunContext(
        world=_StubWorld(),  # type: ignore[arg-type]
        endpoints=endpoints,
        ground_knowledge=GroundKnowledge(estimator=_stub_ground_truth()),
    )


def _make_book(simple_snapshot: NetworkSnapshot) -> UsageBook:
    """Create a UsageBook for the 2-sat fixture."""
    gs = GroundStation("gs1", 0.5, 0.5, "XX", "Test", 8, 25.0, True, 2.1, 1.3, 80.0, False)
    view = CapacityView.from_snapshot(
        sat_state=simple_snapshot.satellite,
        shell=_StubWorld.shell,
        ground_stations={"gs1": gs},
    )
    return UsageBook(view=view)


@pytest.mark.unit
class TestForward:

    def test_basic_flow(
        self, simple_snapshot: NetworkSnapshot, simple_context: RunContext,
    ) -> None:
        plane, cell_grid = _build_routing_plane(simple_snapshot)
        path_table = precompute_path_table(plane, simple_snapshot.satellite.num_sats)
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book, path_table)

        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "google"): 0.01,
        }))
        result = realize(strategy, simple_snapshot, demand, simple_context)

        assert len(result.flow_outcomes) == 1
        flow = result.flow_outcomes[0]
        assert flow.pop_code == "pop"
        assert flow.demand_gbps == 0.01
        assert flow.total_rtt > 0
        # ground_rtt = measurement table one-way (2.5 ms) × 2 = 5 ms
        assert flow.ground_rtt == pytest.approx(5.0)

    def test_total_equals_sum(
        self, simple_snapshot: NetworkSnapshot, simple_context: RunContext,
    ) -> None:
        plane, cell_grid = _build_routing_plane(simple_snapshot)
        path_table = precompute_path_table(plane, simple_snapshot.satellite.num_sats)
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book, path_table)

        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "google"): 0.01,
        }))
        result = realize(strategy, simple_snapshot, demand, simple_context)
        flow = result.flow_outcomes[0]
        expected = flow.satellite_rtt + flow.ground_rtt
        assert abs(flow.total_rtt - expected) < 1e-9

    def test_unknown_dest_unrouted(
        self, simple_snapshot: NetworkSnapshot,
    ) -> None:
        """Flow to unknown destination is unrouted (ground delay KeyError)."""
        ctx = RunContext(
            world=_StubWorld(),  # type: ignore[arg-type]
            endpoints={
                "user_a": Endpoint("user_a", 0.0, 0.0),
                "unknown": Endpoint("unknown", 10.0, 10.0),
            },
            ground_knowledge=GroundKnowledge(estimator=_stub_ground_truth()),
        )
        plane, cell_grid = _build_routing_plane(simple_snapshot)
        path_table = precompute_path_table(plane, simple_snapshot.satellite.num_sats)
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book, path_table)

        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "unknown"): 0.01,
        }))
        result = realize(strategy, simple_snapshot, demand, ctx)
        assert len(result.flow_outcomes) == 0
        assert result.unrouted_demand_gbps > 0
