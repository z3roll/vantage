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
    GatewayAttachments,
    GroundStation,
    GSPoPEdge,
    InfrastructureView,
    ISLEdge,
    ISLGraph,
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
from vantage.forward import RoutingPlaneForward, realize
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

    # Downlink delay is one-way. Sat 1 at (15,15,550) → gs1 at
    # (0.5,0.5,0) gives access_delay ≈ 8.05 ms; using the correct
    # value keeps AccessLink.delay consistent with positions, which
    # compute_egress_options relies on.
    gw = GatewayAttachments(attachments=MappingProxyType({
        "gs1": (AccessLink(sat_id=1, elevation_deg=20.0,
                           slant_range_km=2200.0, delay=8.05),),
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
    # Per-sat Ka feeder cap = 20 Gbps (one antenna per egress beam),
    # matching the documented ShellConfig default and the real
    # CrowdLink spec — was 80 Gbps in earlier WIP, adjusted on
    # 2026-04-17 when the per-sat-feeder reroute path made the cap
    # observable in tests.
    shell = ShellConfig(
        shell_id=1, num_orbits=1, sats_per_orbit=2,
        altitude_km=550.0, inclination_deg=53.0,
        orbit_cycle_s=6000.0, phase_shift=1,
        feeder_capacity_gbps=20.0,
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
        # path_table no longer needed — RoutingPlaneForward computes options lazily
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book)

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
        # path_table no longer needed — RoutingPlaneForward computes options lazily
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book)

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
        # path_table no longer needed — RoutingPlaneForward computes options lazily
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book)

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
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, grid, book)
        ctx = RunContext(
            world=_StubWorld(),  # type: ignore[arg-type]
            endpoints={
                "user_a": Endpoint("user_a", 0.0, 0.0),
                "user_b": Endpoint("user_b", 0.0001, 0.0001),
                "google": Endpoint("google", 37.4, -122.1),
            },
            ground_knowledge=GroundKnowledge(estimator=_stub_ground_truth()),
        )
        # 30 Gbps × 2 = 60 Gbps on a sat_feeder cap of 20 Gbps → ρ ≈ 3.0
        # (the simple_snapshot fixture has only sat 1 attached to gs1, so
        # both flows fall through `charge`'s "all options saturated"
        # branch and end up sharing sat 1 as egress — the multi-egress
        # reroute does not split them onto different sats here, which
        # keeps the same-link premise of this test intact). M/M/1/K
        # queuing differs noticeably between ρ=1.5 (pre-fix flow 1) and
        # ρ=3.0 (pre-fix flow 2); post-fix both should report ρ=3.0
        # queuing identically.
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
        # path_table no longer needed — RoutingPlaneForward computes options lazily
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book)
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
        from vantage.common.link_model import pftk_throughput
        from vantage.forward import effective_throughput

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

    # ─── Bug 14 fix: multi-egress per-sat-feeder reroute ───

    @staticmethod
    def _multi_egress_snapshot() -> NetworkSnapshot:
        """3 sats serving 1 PoP via the same GS.

        - Sat 0 is the source's ingress (overhead at (0, 0)).
        - Sat 1 is the *primary* egress (cheaper ISL hop from sat 0).
        - Sat 2 is the *alternate* egress (longer ISL hop from sat 0).
        - Both sat 1 and sat 2 are attached to gs1, which feeds 'pop'.

        Sat 1 and sat 2 are placed at (15, 15) so they are below the
        source's horizon, leaving sat 0 as the only ingress (matches
        the deterministic visibility convention used by the other
        post-Bug-9 fixtures in this file).
        """
        graph = ISLGraph(
            shell_id=1, timeslot=0, num_sats=3,
            edges=(
                ISLEdge(0, 1, 1.0, 300.0, "intra_orbit"),
                ISLEdge(0, 2, 5.0, 1500.0, "intra_orbit"),
                ISLEdge(1, 2, 0.5, 150.0, "intra_orbit"),
            ),
        )
        positions = np.array([
            [0.0, 0.0, 550.0],
            [15.0, 15.0, 550.0],
            [15.0, 15.0, 550.0],
        ], dtype=np.float64)
        delay_matrix = np.array([
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 0.5],
            [5.0, 0.5, 0.0],
        ])
        # Predecessor matrix (one-hop direct paths, the simple case).
        pred = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ], dtype=np.int32)
        # AccessLink.delay must match position-derived access_delay so
        # compute_egress_options ranks options by realistic costs.
        # Both sats at (15,15,550) → gs1 at (0,0,0): ≈ 8.31 ms
        # one-way. Sat 2 has slightly worse downlink to break the tie
        # so the helper picks sat 1 as primary deterministically (this
        # test specifically asserts the rank order).
        gw = GatewayAttachments(attachments=MappingProxyType({
            "gs1": (
                AccessLink(sat_id=1, elevation_deg=20.0,
                           slant_range_km=2300.0, delay=8.30),
                AccessLink(sat_id=2, elevation_deg=20.0,
                           slant_range_km=2350.0, delay=8.50),
            ),
        }))
        sat = SatelliteState(
            positions=positions, graph=graph,
            delay_matrix=delay_matrix, predecessor_matrix=pred,
            gateway_attachments=gw,
        )
        gs = GroundStation(
            "gs1", 0.0, 0.0, "XX", "Test", 8, 25.0,
            True, 2.1, 1.3, 160.0, False,
        )
        pop = PoP("pop1", "pop", "POP", 0.0, 0.0)
        edge = GSPoPEdge("gs1", "pop", 10.0, 0.05, 100.0)
        infra = InfrastructureView(
            pops=(pop,), ground_stations=(gs,), gs_pop_edges=(edge,),
        )
        return NetworkSnapshot(epoch=0, time_s=0.0, satellite=sat, infra=infra)

    def test_compute_egress_options_returns_top_k_sorted(self) -> None:
        """``compute_egress_options(snapshot, ingress, pop, K)`` returns
        a ranked list of ``EgressOption`` tuples, ascending by
        sat-segment cost. The lowest-cost (egress_sat, gs) is index 0;
        the alternate is index 1; etc."""
        from vantage.forward import compute_egress_options

        snap = self._multi_egress_snapshot()
        opts = compute_egress_options(snap, ingress=0, pop_code="pop", k=8)

        # Two candidates: (egress=1) and (egress=2) — both via gs1.
        assert len(opts) == 2
        assert opts[0].egress_sat == 1, "primary should be sat 1 (cheaper ISL)"
        assert opts[1].egress_sat == 2, "alternate should be sat 2"
        assert opts[0].gs_id == "gs1"
        assert opts[1].gs_id == "gs1"
        assert opts[0].propagation_rtt < opts[1].propagation_rtt

    def test_compute_egress_options_respects_k_limit(self) -> None:
        """Asking for K=1 returns only the primary option."""
        from vantage.forward import compute_egress_options

        snap = self._multi_egress_snapshot()
        opts = compute_egress_options(snap, ingress=0, pop_code="pop", k=1)
        assert len(opts) == 1
        assert opts[0].egress_sat == 1

    def test_charge_reroutes_to_alternate_when_primary_sat_feeder_full(
        self,
    ) -> None:
        """Bug 14 fix: when the primary egress sat's feeder is at
        capacity, the data plane reroutes the flow via an alternate
        egress sat (different ISL hop, different downlink — same GS
        in this fixture)."""
        snap = self._multi_egress_snapshot()
        # Use a custom plane that maps the cell to "pop".
        alice = Endpoint(name="alice", lat_deg=0.0, lon_deg=0.0)
        cell_grid = CellGrid.from_endpoints([("alice", 0.0, 0.0)])
        cell_id = cell_grid.cell_of("alice")
        cell_to_pop = CellToPopTable(
            mapping=MappingProxyType({cell_id: "pop"}),
            version=0, built_at=0.0,
        )
        # Minimal FIB so .decide's legacy fallback path doesn't crash.
        fib_0 = SatelliteFIB(
            sat_id=0,
            fib=MappingProxyType({"pop": FIBEntry.forward(1, cost_ms=5.0)}),
            version=0, built_at=0.0,
        )
        fib_1 = SatelliteFIB(
            sat_id=1,
            fib=MappingProxyType({"pop": FIBEntry.egress("gs1", cost_ms=3.74)}),
            version=0, built_at=0.0,
        )
        fib_2 = SatelliteFIB(
            sat_id=2,
            fib=MappingProxyType({"pop": FIBEntry.egress("gs1", cost_ms=3.78)}),
            version=0, built_at=0.0,
        )
        plane = RoutingPlane(
            cell_to_pop=cell_to_pop,
            sat_fibs=MappingProxyType({0: fib_0, 1: fib_1, 2: fib_2}),
            version=0, built_at=0.0,
        )

        # Pre-fill sat 1's feeder so the next demand can't fit (cap = 20).
        gs = GroundStation(
            "gs1", 0.0, 0.0, "XX", "Test", 8, 25.0,
            True, 2.1, 1.3, 160.0, False,
        )
        view = CapacityView.from_snapshot(
            sat_state=snap.satellite,
            shell=_StubWorld.shell,
            ground_stations={"gs1": gs},
        )
        book = UsageBook(view=view)
        # Sat-feeder cap is 20 Gbps (per ShellConfig default). Pre-load
        # 19 Gbps so a 5 Gbps incoming demand on sat 1 would overflow.
        book.charge_sat_feeder(1, 19.0)

        strategy = RoutingPlaneForward(plane, cell_grid, book)
        ctx = RunContext(
            world=_StubWorld(),  # type: ignore[arg-type]
            endpoints={
                "alice": alice,
                "google": Endpoint("google", 37.4, -122.1),
            },
            ground_knowledge=GroundKnowledge(estimator=_stub_ground_truth()),
        )
        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("alice", "google"): 5.0,
        }))
        result = realize(strategy, snap, demand, ctx)

        assert len(result.flow_outcomes) == 1
        flow = result.flow_outcomes[0]
        # Sat 1 was full (19 + 5 > 20); flow should have rerouted to sat 2.
        assert flow.egress_sat == 2, (
            f"expected reroute to sat 2 (alternate); got egress_sat={flow.egress_sat}"
        )
        # Capacity should reflect: sat 1 still at 19 (untouched by this flow),
        # sat 2 charged with the 5 Gbps.
        assert book.sat_feeder_used[1] == pytest.approx(19.0)
        assert book.sat_feeder_used[2] == pytest.approx(5.0)

    def test_all_options_full_falls_back_to_baseline_pop(self) -> None:
        """When the controller's chosen PoP differs from the cell's
        baseline (geographic-nearest) PoP, ``decide`` appends the
        baseline-pop path as a final fallback option. If every prior
        option's egress sat-feeder is saturated, ``charge`` uses the
        fallback regardless of capacity (overflow accepted; surfaces
        as elevated queuing/loss in measure)."""
        snap = self._multi_egress_snapshot()
        # Add a second PoP with its own GS so baseline can differ
        # from the controller's chosen PoP. Sat 0 itself attaches to
        # this second GS (so baseline path = sat-0-egress, no ISL).
        gs2 = GroundStation(
            "gs2", 0.5, 0.5, "XX", "Test", 8, 25.0,
            True, 2.1, 1.3, 160.0, False,
        )
        pop2 = PoP("pop2", "popB", "POP-B", 0.5, 0.5)
        # Sat 0 at (0, 0, 550) → gs2 at (0.5, 0.5, 0): ≈ 1.86 ms one-way.
        gs2_link = AccessLink(
            sat_id=0, elevation_deg=85.0, slant_range_km=560.0, delay=1.86,
        )
        # Augment the existing snapshot with gs2 + pop2.
        old_attachments = dict(snap.satellite.gateway_attachments.attachments)
        old_attachments["gs2"] = (gs2_link,)
        new_gw = GatewayAttachments(attachments=MappingProxyType(old_attachments))
        new_sat = SatelliteState(
            positions=snap.satellite.positions,
            graph=snap.satellite.graph,
            delay_matrix=snap.satellite.delay_matrix,
            predecessor_matrix=snap.satellite.predecessor_matrix,
            gateway_attachments=new_gw,
        )
        new_infra = InfrastructureView(
            pops=(*snap.infra.pops, pop2),
            ground_stations=(*snap.infra.ground_stations, gs2),
            gs_pop_edges=(
                *snap.infra.gs_pop_edges,
                GSPoPEdge("gs2", "popB", 5.0, 0.02, 100.0),
            ),
        )
        snap = NetworkSnapshot(
            epoch=0, time_s=0.0, satellite=new_sat, infra=new_infra,
        )

        # Cell mapping: baseline = popB (cell's nearest), but controller
        # overrides to "pop" via per_dest. So decide will produce
        # primary options for "pop" + a fallback option for "popB".
        alice = Endpoint(name="alice", lat_deg=0.0, lon_deg=0.0)
        cell_grid = CellGrid.from_endpoints([("alice", 0.0, 0.0)])
        cell_id = cell_grid.cell_of("alice")
        cell_to_pop = CellToPopTable(
            mapping=MappingProxyType({cell_id: "popB"}),  # baseline
            version=0, built_at=0.0,
            per_dest=MappingProxyType({(cell_id, "google"): "pop"}),
        )
        plane = RoutingPlane(
            cell_to_pop=cell_to_pop,
            sat_fibs=MappingProxyType({}),  # FIB unused by new data plane
            version=0, built_at=0.0,
        )

        view = CapacityView.from_snapshot(
            sat_state=snap.satellite,
            shell=_StubWorld.shell,
            ground_stations={"gs1": GroundStation(
                "gs1", 0.0, 0.0, "XX", "Test", 8, 25.0,
                True, 2.1, 1.3, 160.0, False,
            ), "gs2": gs2},
        )
        book = UsageBook(view=view)
        # Saturate BOTH primary egress sats (1 and 2) for "pop" so the
        # fallback to popB (via sat 0) is forced. cap = 20.
        book.charge_sat_feeder(1, 20.0)
        book.charge_sat_feeder(2, 20.0)

        strategy = RoutingPlaneForward(plane, cell_grid, book)
        # Stub ground truth covering BOTH PoPs so decide doesn't
        # short-circuit on missing baseline measurement.
        ctx = RunContext(
            world=_StubWorld(),  # type: ignore[arg-type]
            endpoints={
                "alice": alice,
                "google": Endpoint("google", 37.4, -122.1),
            },
            ground_knowledge=GroundKnowledge(
                estimator=MeasuredGroundDelay(one_way_rtt_ms={
                    ("pop", "google"): 2.5,
                    ("popB", "google"): 1.0,
                }),
            ),
        )
        demand = TrafficDemand(epoch=0, flows=MappingProxyType({
            FlowKey("alice", "google"): 0.5,
        }))
        result = realize(strategy, snap, demand, ctx)

        assert len(result.flow_outcomes) == 1
        flow = result.flow_outcomes[0]
        # Both primary options' sat feeders are full → fallback to
        # popB used. The popB path lands at sat 0 (egress) via gs2.
        assert flow.pop_code == "popB", (
            f"expected fallback to baseline popB; got {flow.pop_code}"
        )
        assert flow.gs_id == "gs2"
        assert flow.egress_sat == 0
        # ground RTT = popB→google measurement (1.0) × 2 = 2.0 ms
        assert flow.ground_rtt == pytest.approx(2.0)

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
        # path_table no longer needed — RoutingPlaneForward computes options lazily
        book = _make_book(simple_snapshot)
        strategy = RoutingPlaneForward(plane, cell_grid, book)
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
