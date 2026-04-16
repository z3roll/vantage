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
    # Sat 1 is intentionally placed far from sat 0 (below the source's
    # horizon) so the source at (0, 0) only ever has sat 0 as a visible
    # ingress candidate. This keeps the per-realize stochastic
    # `find_ingress_satellite` deterministic for tests without having
    # to monkey-patch RNG state.
    positions = np.array([[0.0, 0.0, 550.0], [15.0, 15.0, 550.0]])
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


# ---------------------------------------------------------------------------
# Audit fixes (2026-04-17): bugs A/B/C/D from forward-pass audit.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestForwardAuditFixes:
    """Regression tests for the four critical bugs found by re-auditing
    the queuing-aware forward path. Each test name ties back to the
    audit summary in the chat."""

    def _multi_user_grid(self) -> tuple[CellGrid, RoutingPlane]:
        """Build a grid + plane where user_a and user_b live in the same
        H3 cell so both flows take the identical sat path."""
        grid = CellGrid.from_endpoints([
            ("user_a", 0.0, 0.0),
            ("user_b", 0.0001, 0.0001),
        ])
        cell_id = grid.cell_of("user_a")
        assert cell_id == grid.cell_of("user_b"), (
            "fixture invariant: both users must hash to the same cell"
        )
        cell_to_pop = CellToPopTable(
            mapping=MappingProxyType({cell_id: "pop"}),
            version=0, built_at=0.0,
        )
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
        plane = RoutingPlane(
            cell_to_pop=cell_to_pop,
            sat_fibs=MappingProxyType({0: fib_0, 1: fib_1}),
            version=0, built_at=0.0,
        )
        return grid, plane

    # --- Bug A ---

    def test_queuing_is_order_independent_for_flows_on_same_link(
        self, simple_snapshot: NetworkSnapshot, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bug A: two flows on the same link must report the same
        queuing delay. Pre-fix, ``RoutingPlaneForward.resolve_flow``
        charged BEFORE measuring, so the first flow saw ρ from itself
        only and the second saw the aggregate. Post-fix, ``realize``
        runs in two passes and both flows measure against the final
        per-link load.

        Monkey-patches ``find_ingress_satellite`` to deterministically
        return sat 0 for every source — otherwise the simple 2-sat
        fixture's stochastic 80/20 branch can scatter the two srcs
        onto different ingress sats (and therefore different paths),
        defeating the same-link premise of the test."""
        import vantage.forward as fwd
        from vantage.domain import AccessLink as _AL

        def _always_sat0(src: object, sat_positions: object, **_: object) -> _AL:
            return _AL(sat_id=0, elevation_deg=89.0, slant_range_km=550.0, delay=1.83)

        monkeypatch.setattr(fwd, "find_ingress_satellite", _always_sat0)

        grid, plane = self._multi_user_grid()
        path_table = precompute_path_table(plane, simple_snapshot.satellite.num_sats)
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, grid, book, path_table)
        ctx = RunContext(
            world=_StubWorld(),  # type: ignore[arg-type]
            endpoints={
                "user_a": Endpoint("user_a", 0.0, 0.0),
                "user_b": Endpoint("user_b", 0.0001, 0.0001),
                "google": Endpoint("google", 37.4, -122.1),
            },
            ground_knowledge=GroundKnowledge(estimator=_stub_ground_truth()),
        )
        # 30 Gbps × 2 = 60 Gbps on a sat_feeder cap of 80 Gbps → ρ ≈ 0.75,
        # large enough that M/M/1/K queuing differs noticeably between
        # ρ=0.375 (pre-fix flow 1) and ρ=0.75 (pre-fix flow 2).
        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "google"): 30.0,
            FlowKey("user_b", "google"): 30.0,
        }))
        result = realize(strategy, simple_snapshot, demand, ctx)

        assert len(result.flow_outcomes) == 2
        # Sanity: both flows must take the same path (same user_sat,
        # so same isl_links + same egress).
        user_sats = {fo.user_sat for fo in result.flow_outcomes}
        assert user_sats == {0}, (
            f"fixture invariant: both flows must share ingress sat 0; got {user_sats}"
        )
        q0 = result.flow_outcomes[0].queuing_rtt
        q1 = result.flow_outcomes[1].queuing_rtt
        assert q0 == pytest.approx(q1, rel=1e-9), (
            f"queuing_rtt depends on iteration order — Bug A not fixed: "
            f"q0={q0:.6e} q1={q1:.6e}"
        )
        assert q0 > 0, (
            f"expected non-zero queuing at ρ ≈ 0.75 (sanity check that "
            f"the test is actually exercising the M/M/1/K branch); got {q0}"
        )

    # --- Bug B ---

    def test_failed_ground_lookup_does_not_charge_capacity(
        self, simple_snapshot: NetworkSnapshot,
    ) -> None:
        """Bug B: when the ground-delay estimator can't serve a (pop,
        dest) pair, the flow must be unrouted AND no capacity gets
        charged. Pre-fix, ``resolve_flow`` charged isl/sat_feeder/
        gs_feeder before checking ground_truth, so dropped flows
        polluted the usage book and over-stated link utilization."""
        plane, cell_grid = _build_routing_plane(simple_snapshot)
        path_table = precompute_path_table(plane, simple_snapshot.satellite.num_sats)
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book, path_table)
        # The stub ground table only knows ("pop", "google"). "void" is
        # a registered endpoint with no measurement.
        ctx = RunContext(
            world=_StubWorld(),  # type: ignore[arg-type]
            endpoints={
                "user_a": Endpoint("user_a", 0.0, 0.0),
                "void": Endpoint("void", 10.0, 10.0),
            },
            ground_knowledge=GroundKnowledge(estimator=_stub_ground_truth()),
        )
        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "void"): 0.5,
        }))
        result = realize(strategy, simple_snapshot, demand, ctx)

        assert len(result.flow_outcomes) == 0
        assert result.unrouted_demand_gbps == pytest.approx(0.5)
        assert dict(book.isl_used) == {}, (
            f"ISL charged despite drop — Bug B not fixed: "
            f"{dict(book.isl_used)}"
        )
        assert dict(book.sat_feeder_used) == {}, (
            f"sat_feeder charged despite drop — Bug B not fixed: "
            f"{dict(book.sat_feeder_used)}"
        )
        assert dict(book.gs_feeder_used) == {}, (
            f"gs_feeder charged despite drop — Bug B not fixed: "
            f"{dict(book.gs_feeder_used)}"
        )

    # --- Bug C ---

    def testeffective_throughput_capped_by_bottleneck_under_loss(self) -> None:
        """Bug C: the loss-branch returned ``min(demand, pftk)`` and
        ignored ``bottleneck_gbps``. With low-but-nonzero loss and
        modest RTT, PFTK can return values much larger than the
        physical link capacity, so ``effective_throughput_gbps`` ends
        up over-stated. Tested at the helper level because triggering
        the regime through a full ``realize`` requires tiny buffers
        the M/M/1/K default does not exercise."""
        from vantage.forward import effective_throughput
        from vantage.common.link_model import pftk_throughput

        # Sanity: at low loss + low RTT PFTK far exceeds a 50 Mbps cap.
        pftk = pftk_throughput(10.0, 1e-6)
        assert pftk > 0.05, (
            f"fixture invariant: PFTK should exceed 50 Mbps in this regime; got {pftk}"
        )

        eff = effective_throughput(
            demand_gbps=10.0,
            total_rtt_ms=10.0,
            loss_probability=1e-6,
            bottleneck_gbps=0.05,  # 50 Mbps bottleneck
        )
        assert eff <= 0.05 + 1e-9, (
            f"eff_tput {eff:.6f} exceeds bottleneck 0.05 Gbps — Bug C not fixed"
        )

        # Also verify the no-loss branch still respects bottleneck
        # (regression for the existing else-branch).
        eff_noloss = effective_throughput(
            demand_gbps=10.0, total_rtt_ms=10.0,
            loss_probability=0.0, bottleneck_gbps=0.05,
        )
        assert eff_noloss == pytest.approx(0.05)

    # --- Bug D ---

    def test_uplink_cached_per_src_per_epoch(
        self, simple_snapshot: NetworkSnapshot, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bug D: each flow re-called ``find_ingress_satellite`` with a
        stochastic 80/20 branch over a shared module RNG. A single
        terminal could be assigned different ingress sats across its
        flows in the same epoch. Post-fix: ``realize`` caches the
        uplink per-src per-epoch, so multiple flows from one source
        share the same ingress."""
        plane, cell_grid = _build_routing_plane(simple_snapshot)
        path_table = precompute_path_table(plane, simple_snapshot.satellite.num_sats)
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book, path_table)
        ctx = RunContext(
            world=_StubWorld(),  # type: ignore[arg-type]
            endpoints={
                "user_a": Endpoint("user_a", 0.0, 0.0),
                "google": Endpoint("google", 37.4, -122.1),
                "facebook": Endpoint("facebook", 37.4, -122.1),
            },
            ground_knowledge=GroundKnowledge(
                estimator=MeasuredGroundDelay(one_way_rtt_ms={
                    ("pop", "google"): 2.5,
                    ("pop", "facebook"): 3.5,
                }),
            ),
        )

        import vantage.forward as fwd
        call_count = [0]
        real_fn = fwd.find_ingress_satellite

        def counting_wrapper(*args: object, **kwargs: object) -> object:
            call_count[0] += 1
            return real_fn(*args, **kwargs)

        monkeypatch.setattr(fwd, "find_ingress_satellite", counting_wrapper)

        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("user_a", "google"): 0.01,
            FlowKey("user_a", "facebook"): 0.01,
        }))
        result = realize(strategy, simple_snapshot, demand, ctx)

        assert len(result.flow_outcomes) == 2
        assert call_count[0] == 1, (
            f"find_ingress_satellite called {call_count[0]} times for one "
            f"source — Bug D not fixed (must be cached per-src per-epoch)"
        )
        user_sats = {fo.user_sat for fo in result.flow_outcomes}
        assert user_sats == {0}, (
            f"expected all flows from user_a to share ingress sat 0; got {user_sats}"
        )
