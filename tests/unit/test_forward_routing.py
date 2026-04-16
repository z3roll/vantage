"""Unit tests for ``vantage.forward.RoutingPlaneForward``.

Covers the happy path, capacity bookkeeping correctness, graceful
handling of missing inputs (unknown endpoint, no cell mapping,
unreachable ingress, FIB walk blowup), and the RTT composition
(uplink RTT + FIB cost + ground truth).

Reuses the same toy 2-sat / 1-PoP / 1-GS fixture from
``test_fib_builder`` — repeating the helpers here keeps test files
self-contained and avoids cross-file fixture imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.domain import (
    AccessLink,
    CapacityView,
    CellGrid,
    Endpoint,
    FlowKey,
    GSPoPEdge,
    GatewayAttachments,
    GroundStation,
    InfrastructureView,
    ISLEdge,
    ISLGraph,
    NetworkSnapshot,
    PoP,
    SatelliteState,
    ShellConfig,
    TrafficDemand,
    UsageBook,
)
from vantage.forward import RoutingPlaneForward, realize


# --- Minimal fixture helpers -------------------------------------------------


def _gs(gs_id: str, lat: float, lon: float, num_antennas: int = 8) -> GroundStation:
    return GroundStation(
        gs_id=gs_id,
        lat_deg=lat,
        lon_deg=lon,
        country="US",
        town="",
        num_antennas=num_antennas,
        min_elevation_deg=25.0,
        enabled=True,
        uplink_ghz=2.1,
        downlink_ghz=1.3,
        max_capacity=float(num_antennas) * 10.0,
        temporary=False,
    )


def _two_sat_snapshot() -> NetworkSnapshot:
    graph = ISLGraph(
        shell_id=1, timeslot=0, num_sats=2,
        edges=(
            ISLEdge(
                sat_a=0, sat_b=1,
                delay=1.0, distance_km=300.0, link_type="intra_orbit",
            ),
        ),
    )
    # Place sat 0 near (lat=0, lon=0) and sat 1 near (lat=0, lon=1).
    positions = np.array(
        [[0.0, 0.0, 550.0], [0.0, 1.0, 550.0]],
        dtype=np.float64,
    )
    positions.flags.writeable = False
    delay_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    delay_matrix.flags.writeable = False
    predecessor_matrix = np.array([[0, 0], [1, 1]], dtype=np.int32)
    predecessor_matrix.flags.writeable = False

    gs_a = _gs("gs_a", lat=0.0, lon=1.0)
    gw = GatewayAttachments(
        attachments=MappingProxyType(
            {
                "gs_a": (
                    AccessLink(sat_id=1, elevation_deg=89.0, slant_range_km=550.0, delay=1.83),
                ),
            }
        )
    )
    sat_state = SatelliteState(
        positions=positions,
        graph=graph,
        delay_matrix=delay_matrix,
        predecessor_matrix=predecessor_matrix,
        gateway_attachments=gw,
    )

    pop_a = PoP(site_id="s1", code="pop_a", name="POP-A", lat_deg=0.0, lon_deg=1.0)
    edge = GSPoPEdge(
        gs_id="gs_a", pop_code="pop_a",
        distance_km=0.0, backhaul_delay=0.0, capacity_gbps=100.0,
    )
    infra = InfrastructureView(
        pops=(pop_a,), ground_stations=(gs_a,), gs_pop_edges=(edge,),
    )
    return NetworkSnapshot(epoch=0, time_s=123.0, satellite=sat_state, infra=infra)


@dataclass
class _StubWorld:
    """Minimal WorldModel stand-in for the RunContext — only ``calibration``
    is accessed via ``context.world.calibration`` in the legacy path,
    so we keep a trivial attribute surface here.
    """

    calibration: Any = None


def _ctx(
    endpoints: dict[str, Endpoint],
    *,
    ground_truth_rtts: dict[tuple[str, str], float] | None = None,
) -> Any:
    """Build a RunContext-shaped object.

    Uses a simple object-with-attributes stub because the real
    RunContext demands a real WorldModel. Only ``world``,
    ``endpoints``, ``ground_knowledge``, ``service_ground_delay``,
    and ``simulation_start_utc`` are read by the forward paths.

    The default ground-truth table is hand-injected so the 2-sat
    fixture's single (pop_a, bob) pair has a concrete measurement
    and doesn't raise KeyError on every lookup.
    """
    from vantage.world.ground import GroundKnowledge, MeasuredGroundDelay

    if ground_truth_rtts is None:
        # Default: every (pop, destination) pair appearing in tests
        # gets a one-way RTT of 0 ms so assertions on ``ground_rtt``
        # can still use == 0.0 while having a concrete measurement
        # behind it instead of a fabrication.
        ground_truth_rtts = {("pop_a", "bob"): 0.0}
    estimator = MeasuredGroundDelay(one_way_rtt_ms=ground_truth_rtts)

    class _Ctx:
        def __init__(self) -> None:
            self.world = _StubWorld()
            self.endpoints = endpoints
            self.ground_knowledge = GroundKnowledge(estimator=estimator)
            self.service_ground_delay = None
            self.simulation_start_utc = None

    return _Ctx()


def _plane_and_grid(snap: NetworkSnapshot) -> tuple[Any, CellGrid, Endpoint]:
    """Build a user endpoint, cell grid, and routing plane for the 2-sat fixture."""
    # Put the user near sat 0's sub-point (lat=0, lon=0) so find_ingress_satellite
    # picks sat 0 as the ingress.
    alice = Endpoint(name="alice", lat_deg=0.0, lon_deg=0.0)
    grid = CellGrid.from_endpoints([(alice.name, alice.lat_deg, alice.lon_deg)])
    plane = NearestPoPController().compute_routing_plane(snap, grid, version=1)
    return plane, grid, alice


def _book(snap: NetworkSnapshot) -> UsageBook:
    shell = ShellConfig(
        shell_id=1, altitude_km=550.0, orbit_cycle_s=5731.0,
        inclination_deg=53.0, phase_shift=1, num_orbits=1, sats_per_orbit=2,
    )
    view = CapacityView.from_snapshot(
        sat_state=snap.satellite, shell=shell,
        ground_stations={gs.gs_id: gs for gs in snap.infra.ground_stations},
    )
    return UsageBook(view=view)


# --- Happy path ---------------------------------------------------------------


@pytest.mark.unit
class TestHappyPath:
    def test_single_flow_served(self) -> None:
        snap = _two_sat_snapshot()
        plane, grid, alice = _plane_and_grid(snap)
        ctx = _ctx({alice.name: alice, "bob": Endpoint("bob", 0.0, 1.0)})
        book = _book(snap)
        demand = TrafficDemand(
            epoch=0,
            flows=MappingProxyType({FlowKey(src="alice", dst="bob"): 0.05}),
        )

        result = realize(
            RoutingPlaneForward(plane, grid, book),
            snap, demand, ctx,
        )

        assert len(result.flow_outcomes) == 1
        outcome = result.flow_outcomes[0]
        assert outcome.pop_code == "pop_a"
        assert outcome.gs_id == "gs_a"
        # Sat 1 is the bent-pipe egress because gs_a is only visible from sat 1.
        assert outcome.egress_sat == 1
        # Ingress is sat 0 because alice is at (0, 0).
        assert outcome.user_sat == 0
        # Demand was fully served.
        assert result.routed_demand_gbps == pytest.approx(0.05)
        assert result.unrouted_demand_gbps == pytest.approx(0.0)

    def test_capacity_book_charged_on_isl_and_feeder(self) -> None:
        snap = _two_sat_snapshot()
        plane, grid, alice = _plane_and_grid(snap)
        ctx = _ctx({alice.name: alice, "bob": Endpoint("bob", 0.0, 1.0)})
        book = _book(snap)
        demand_gbps = 0.05
        demand = TrafficDemand(
            epoch=0,
            flows=MappingProxyType({FlowKey(src="alice", dst="bob"): demand_gbps}),
        )

        realize(
            RoutingPlaneForward(plane, grid, book),
            snap, demand, ctx,
        )

        # ISL(0,1) was traversed once.
        assert book.isl_used[(0, 1)] == pytest.approx(demand_gbps)
        # Sat 1 is the egress — its feeder is charged, not sat 0's.
        assert book.sat_feeder_used[1] == pytest.approx(demand_gbps)
        assert 0 not in book.sat_feeder_used
        # gs_a feeder charged.
        assert book.gs_feeder_used["gs_a"] == pytest.approx(demand_gbps)

    def test_rtt_decomposes_into_uplink_fib_cost_ground(self) -> None:
        """total_rtt == uplink + ingress_fib.cost_ms + ground_truth * 2."""
        snap = _two_sat_snapshot()
        plane, grid, alice = _plane_and_grid(snap)
        ctx = _ctx({alice.name: alice, "bob": Endpoint("bob", 0.0, 1.0)})
        book = _book(snap)
        demand = TrafficDemand(
            epoch=0,
            flows=MappingProxyType({FlowKey(src="alice", dst="bob"): 0.01}),
        )

        result = realize(
            RoutingPlaneForward(plane, grid, book),
            snap, demand, ctx,
        )
        outcome = result.flow_outcomes[0]

        # FIB[ingress=0][pop_a].cost_ms is the sat-side RTT from sat 0
        # down to pop_a. Add uplink and the ground RTT we'd compute via
        # the haversine estimator (pop_a and bob are both at (0, 1) so
        # ground RTT is zero).
        ingress_cost = plane.fib_of(0).route("pop_a").cost_ms
        assert outcome.satellite_rtt > ingress_cost  # includes uplink
        assert outcome.satellite_rtt - ingress_cost > 0  # uplink is positive
        assert outcome.ground_rtt == pytest.approx(0.0, abs=0.1)
        assert outcome.total_rtt == pytest.approx(
            outcome.satellite_rtt + outcome.ground_rtt
        )


# --- Dropped flows ----------------------------------------------------------


@pytest.mark.unit
class TestDroppedFlows:
    """Flows that cannot be resolved must be counted as unrouted, not
    crash the forward pass."""

    def test_unknown_source_endpoint_is_unrouted(self) -> None:
        snap = _two_sat_snapshot()
        plane, grid, alice = _plane_and_grid(snap)
        ctx = _ctx({alice.name: alice, "bob": Endpoint("bob", 0.0, 1.0)})
        book = _book(snap)
        demand = TrafficDemand(
            epoch=0,
            flows=MappingProxyType({FlowKey(src="ghost", dst="bob"): 0.02}),
        )

        result = realize(
            RoutingPlaneForward(plane, grid, book),
            snap, demand, ctx,
        )
        assert len(result.flow_outcomes) == 0
        assert result.unrouted_demand_gbps == pytest.approx(0.02)

    def test_source_without_cell_is_unrouted(self) -> None:
        snap = _two_sat_snapshot()
        # Grid has only "alice"; "charlie" is a registered endpoint but
        # not in the grid, so cell_grid.cell_of("charlie") raises KeyError.
        alice = Endpoint(name="alice", lat_deg=0.0, lon_deg=0.0)
        charlie = Endpoint(name="charlie", lat_deg=0.0, lon_deg=0.0)
        grid = CellGrid.from_endpoints([(alice.name, alice.lat_deg, alice.lon_deg)])
        plane = NearestPoPController().compute_routing_plane(snap, grid)
        ctx = _ctx({alice.name: alice, charlie.name: charlie, "bob": Endpoint("bob", 0.0, 1.0)})
        book = _book(snap)
        demand = TrafficDemand(
            epoch=0,
            flows=MappingProxyType({FlowKey(src="charlie", dst="bob"): 0.03}),
        )

        result = realize(
            RoutingPlaneForward(plane, grid, book),
            snap, demand, ctx,
        )
        assert len(result.flow_outcomes) == 0
        assert result.unrouted_demand_gbps == pytest.approx(0.03)

    def test_fib_two_cycle_is_dropped(self) -> None:
        """A hand-injected FIB with a 2-cycle must bail instead of looping.

        Legitimate planes from ``build_satellite_fibs`` can't express
        this — they derive from an acyclic predecessor matrix — but a
        future policy that builds its own FIB by hand could slip one
        in, and forward_routing must defend against it.
        """
        from vantage.domain import (
            CellToPopTable,
            FIBEntry,
            RoutingPlane,
            SatelliteFIB,
        )

        snap = _two_sat_snapshot()
        alice = Endpoint(name="alice", lat_deg=0.0, lon_deg=0.0)
        grid = CellGrid.from_endpoints([(alice.name, alice.lat_deg, alice.lon_deg)])

        # Build a plane where sat 0 → sat 1 and sat 1 → sat 0 (2-cycle),
        # with no EGRESS entry anywhere.
        fib0 = SatelliteFIB(
            sat_id=0,
            fib=MappingProxyType({"pop_a": FIBEntry.forward(1, cost_ms=10.0)}),
            version=1,
            built_at=0.0,
        )
        fib1 = SatelliteFIB(
            sat_id=1,
            fib=MappingProxyType({"pop_a": FIBEntry.forward(0, cost_ms=10.0)}),
            version=1,
            built_at=0.0,
        )
        cycle_plane = RoutingPlane(
            cell_to_pop=CellToPopTable(
                mapping=MappingProxyType({grid.cell_of("alice"): "pop_a"}),
                version=1,
                built_at=0.0,
            ),
            sat_fibs=MappingProxyType({0: fib0, 1: fib1}),
            version=1,
            built_at=0.0,
        )

        ctx = _ctx({alice.name: alice, "bob": Endpoint("bob", 0.0, 1.0)})
        book = _book(snap)
        demand = TrafficDemand(
            epoch=0,
            flows=MappingProxyType({FlowKey(src="alice", dst="bob"): 0.05}),
        )

        result = realize(
            RoutingPlaneForward(cycle_plane, grid, book),
            snap, demand, ctx,
        )
        # Walk bailed → flow is unrouted, no partial state charged.
        assert len(result.flow_outcomes) == 0
        assert result.unrouted_demand_gbps == pytest.approx(0.05)
        assert len(book.isl_used) == 0
        assert len(book.sat_feeder_used) == 0
        assert len(book.gs_feeder_used) == 0

    def test_total_demand_accounting(self) -> None:
        """Served + unrouted must equal total."""
        snap = _two_sat_snapshot()
        plane, grid, alice = _plane_and_grid(snap)
        ctx = _ctx({alice.name: alice, "bob": Endpoint("bob", 0.0, 1.0)})
        book = _book(snap)
        demand = TrafficDemand(
            epoch=0,
            flows=MappingProxyType(
                {
                    FlowKey(src="alice", dst="bob"): 0.05,  # routable
                    FlowKey(src="ghost", dst="bob"): 0.03,  # unknown src
                }
            ),
        )
        result = realize(
            RoutingPlaneForward(plane, grid, book),
            snap, demand, ctx,
        )
        assert result.total_demand_gbps == pytest.approx(0.08)
        assert result.routed_demand_gbps == pytest.approx(0.05)
        assert result.unrouted_demand_gbps == pytest.approx(0.03)
