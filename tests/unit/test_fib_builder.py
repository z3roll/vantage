"""Unit tests for per-sat routing, FIB construction, and the
nearest-PoP baseline routing plane.

Covers:

    * :func:`precompute_per_sat_routing` — argmin bookkeeping
    * :func:`build_satellite_fibs` — EGRESS vs FORWARD variants, skip
      unreachable, next-hop derivation via predecessor walk
    * :func:`build_cell_to_pop_nearest` — geographic argmin, empty PoPs
    * :func:`build_routing_plane_nearest_pop` — end-to-end composition
    * :meth:`NearestPoPController.compute_routing_plane` — plumbing
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from vantage.common import access_delay
from vantage.control.policy.common.fib_builder import (
    build_cell_to_pop_nearest,
    build_routing_plane_nearest_pop,
    build_satellite_fibs,
)
from vantage.control.policy.common.sat_cost import (
    PerSatRouting,
    precompute_per_sat_routing,
)
from vantage.control.policy.nearest_pop import NearestPoPController
from vantage.domain import (
    AccessLink,
    CellGrid,
    GSPoPEdge,
    GatewayAttachments,
    GroundStation,
    InfrastructureView,
    ISLEdge,
    ISLGraph,
    NetworkSnapshot,
    PoP,
    SatelliteState,
    cell_id_to_str,
    latlng_to_cell_id,
)


# --- Expected RTT helpers ---------------------------------------------------


def _expected_downlink_rtt_sat1_to_gs_a() -> float:
    """Downlink RTT (ms) from sat 1 to gs_a in the 2-sat fixture.

    sat 1 is at (lat=0, lon=1, alt=550); gs_a is at (lat=0, lon=1, alt=0).
    The slant range is dominated by the 550 km altitude, not the zero
    horizontal offset. Computed via the same helper the production code
    uses so the expected values track any future geometry tweak.
    """
    return access_delay(
        0.0, 1.0,
        0.0, 1.0, 550.0,
    ) * 2


# --- Minimal fixture builders ------------------------------------------------


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
    """Minimal 2-sat, 1-PoP, 1-GS snapshot.

    Topology:
        sat 0 ── ISL (1 ms one-way) ── sat 1
        gs_a visible only from sat 1 (bent-pipe)
        pop_a reachable via (gs_a) with 0 backhaul

    Expected nearest-PoP routing:
        ingress 0 → FORWARD to next_hop=1 (cost 2 ms RTT via ISL)
        ingress 1 → EGRESS to gs_a     (cost 0 ms)
    """
    # ISL graph: single edge 0 ↔ 1
    graph = ISLGraph(
        shell_id=1,
        timeslot=0,
        num_sats=2,
        edges=(
            ISLEdge(
                sat_a=0, sat_b=1,
                delay=1.0, distance_km=300.0, link_type="intra_orbit",
            ),
        ),
    )

    positions = np.array(
        [
            [0.0, 0.0, 550.0],
            [0.0, 1.0, 550.0],
        ],
        dtype=np.float64,
    )
    positions.flags.writeable = False

    delay_matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    delay_matrix.flags.writeable = False

    # predecessor_matrix[src, dst] = second-to-last node on path src→dst.
    # Diagonal contains src (per compute_all_pairs contract).
    predecessor_matrix = np.array(
        [
            [0, 0],  # 0→0: self; 0→1: prev=0 (direct hop)
            [1, 1],  # 1→0: prev=1 (direct hop); 1→1: self
        ],
        dtype=np.int32,
    )
    predecessor_matrix.flags.writeable = False

    gs_links = GatewayAttachments(
        attachments=MappingProxyType(
            {
                "gs_a": (
                    AccessLink(sat_id=1, elevation_deg=89.0, slant_range_km=1.0, delay=0.0),
                ),
            }
        )
    )

    sat_state = SatelliteState(
        positions=positions,
        graph=graph,
        delay_matrix=delay_matrix,
        predecessor_matrix=predecessor_matrix,
        gateway_attachments=gs_links,
    )

    pop_a = PoP(site_id="s1", code="pop_a", name="POP-A", lat_deg=0.0, lon_deg=1.0)
    gs_a = _gs("gs_a", lat=0.0, lon=1.0)
    edge = GSPoPEdge(
        gs_id="gs_a", pop_code="pop_a",
        distance_km=0.0, backhaul_delay=0.0, capacity_gbps=100.0,
    )
    infra = InfrastructureView(
        pops=(pop_a,),
        ground_stations=(gs_a,),
        gs_pop_edges=(edge,),
    )

    return NetworkSnapshot(epoch=0, time_s=123.0, satellite=sat_state, infra=infra)


# --- precompute_per_sat_routing ---------------------------------------------


@pytest.mark.unit
class TestPrecomputePerSatRouting:
    def test_reachability_both_sats(self) -> None:
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)
        assert routing.is_reachable("pop_a", 0)
        assert routing.is_reachable("pop_a", 1)

    def test_egress_argmin_local_sat(self) -> None:
        """Sat 1 is itself the egress for pop_a (bent-pipe)."""
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)
        assert int(routing.egress_sat["pop_a"][1]) == 1
        assert routing.chosen_gs("pop_a", 1) == "gs_a"

    def test_egress_argmin_remote_sat(self) -> None:
        """From sat 0, the only egress to pop_a is sat 1."""
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)
        assert int(routing.egress_sat["pop_a"][0]) == 1
        assert routing.chosen_gs("pop_a", 0) == "gs_a"

    def test_cost_rtt_from_remote_includes_isl_and_downlink(self) -> None:
        """Cost from sat 0 = ISL RTT + downlink RTT (backhaul is 0)."""
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)
        expected = 2.0 + _expected_downlink_rtt_sat1_to_gs_a()
        assert routing.cost_ms["pop_a"][0] == pytest.approx(expected)

    def test_cost_rtt_from_local_is_downlink_only(self) -> None:
        """Sat 1 is the egress — only downlink + backhaul, no ISL."""
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)
        expected = _expected_downlink_rtt_sat1_to_gs_a()
        assert routing.cost_ms["pop_a"][1] == pytest.approx(expected)

    def test_result_is_immutable(self) -> None:
        """PerSatRouting freezes its dict containers and underlying arrays."""
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)
        # Dict containers are MappingProxyType — no item assignment.
        with pytest.raises(TypeError):
            routing.cost_ms["pop_a"] = np.zeros(2)  # type: ignore[index]
        # Arrays are read-only — no element assignment.
        with pytest.raises(ValueError, match="read-only"):
            routing.cost_ms["pop_a"][0] = 0.0
        with pytest.raises(ValueError, match="read-only"):
            routing.egress_sat["pop_a"][0] = 0

    def test_unreachable_pop_marked(self) -> None:
        """If no GS serves the PoP, every ingress reports unreachable."""
        snap = _two_sat_snapshot()
        # Strip the GS attachment so the PoP is unreachable.
        stripped_sat = SatelliteState(
            positions=snap.satellite.positions,
            graph=snap.satellite.graph,
            delay_matrix=snap.satellite.delay_matrix,
            predecessor_matrix=snap.satellite.predecessor_matrix,
            gateway_attachments=GatewayAttachments(attachments=MappingProxyType({})),
        )
        snap = NetworkSnapshot(
            epoch=snap.epoch, time_s=snap.time_s,
            satellite=stripped_sat, infra=snap.infra,
        )
        routing = precompute_per_sat_routing(snap)
        assert not routing.is_reachable("pop_a", 0)
        assert not routing.is_reachable("pop_a", 1)
        assert int(routing.egress_sat["pop_a"][0]) == -1


# --- build_satellite_fibs ---------------------------------------------------


@pytest.mark.unit
class TestBuildSatelliteFibs:
    def test_egress_entry_for_local_sat(self) -> None:
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)
        fibs = build_satellite_fibs(snap, routing, version=7)
        entry = fibs[1].route("pop_a")
        assert entry.is_egress
        assert entry.egress_gs == "gs_a"
        assert entry.cost_ms == pytest.approx(_expected_downlink_rtt_sat1_to_gs_a())

    def test_forward_entry_for_remote_sat(self) -> None:
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)
        fibs = build_satellite_fibs(snap, routing, version=7)
        entry = fibs[0].route("pop_a")
        assert entry.is_forward
        assert entry.next_hop_sat == 1
        assert entry.cost_ms == pytest.approx(
            2.0 + _expected_downlink_rtt_sat1_to_gs_a()
        )

    def test_fib_version_and_timestamp_propagated(self) -> None:
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)
        fibs = build_satellite_fibs(snap, routing, version=42)
        for fib in fibs.values():
            assert fib.version == 42
            assert fib.built_at == snap.time_s

    def test_unreachable_pop_absent_from_fib(self) -> None:
        """Marking ingress 0 unreachable for pop_a must drop its FIB entry.

        PerSatRouting is immutable, so we build a fresh instance with
        writable copies of the arrays (copies drop the read-only flag),
        mutate the copies, then hand them to the new PerSatRouting
        constructor which re-freezes.
        """
        snap = _two_sat_snapshot()
        routing = precompute_per_sat_routing(snap)

        cost_ms = {k: v.copy() for k, v in routing.cost_ms.items()}
        egress_sat = {k: v.copy() for k, v in routing.egress_sat.items()}
        gs_index = {k: v.copy() for k, v in routing.gs_index.items()}
        cost_ms["pop_a"][0] = float("inf")
        egress_sat["pop_a"][0] = -1
        gs_index["pop_a"][0] = -1
        corrupted = PerSatRouting(
            cost_ms=cost_ms,
            egress_sat=egress_sat,
            gs_index=gs_index,
            gs_ids=routing.gs_ids,
        )

        fibs = build_satellite_fibs(snap, corrupted)
        assert "pop_a" not in fibs[0].fib  # skipped, no entry
        assert "pop_a" in fibs[1].fib  # sat 1 still reachable


# --- build_cell_to_pop_nearest ----------------------------------------------


@pytest.mark.unit
class TestBuildCellToPopNearest:
    def test_each_cell_assigned_to_nearest_pop(self) -> None:
        # Two PoPs: Tokyo and Seattle.
        pops = (
            PoP(site_id="tk", code="tok", name="TOK", lat_deg=35.68, lon_deg=139.77),
            PoP(site_id="sea", code="sea", name="SEA", lat_deg=47.60, lon_deg=-122.33),
        )
        # Two endpoints: one near Tokyo, one near Seattle.
        grid = CellGrid.from_endpoints(
            [
                ("tokyo_user", 35.70, 139.80),
                ("seattle_user", 47.60, -122.30),
            ]
        )
        table = build_cell_to_pop_nearest(grid, pops, built_at=0.0)

        tokyo_cell = grid.cell_of("tokyo_user")
        seattle_cell = grid.cell_of("seattle_user")
        assert table.pop_of(tokyo_cell) == "tok"
        assert table.pop_of(seattle_cell) == "sea"

    def test_returns_ranked_top_n_cascade(self) -> None:
        """Each cell gets a tuple of up to top_n PoPs sorted by distance ASC."""
        pops = (
            PoP(site_id="tk", code="tok", name="TOK", lat_deg=35.68, lon_deg=139.77),
            PoP(site_id="sea", code="sea", name="SEA", lat_deg=47.60, lon_deg=-122.33),
            PoP(site_id="lhr", code="lhr", name="LHR", lat_deg=51.50, lon_deg=-0.13),
            PoP(site_id="syd", code="syd", name="SYD", lat_deg=-33.87, lon_deg=151.21),
        )
        grid = CellGrid.from_endpoints([("tokyo_user", 35.70, 139.80)])
        table = build_cell_to_pop_nearest(grid, pops, built_at=0.0, top_n=3)

        cascade = table.pops_of(grid.cell_of("tokyo_user"))
        # Length capped at top_n; sorted by distance from Tokyo.
        assert len(cascade) == 3
        assert cascade[0] == "tok"  # closest
        # Sydney is the second-closest of the four to Tokyo (~7800 km),
        # then Seattle (~7700 km), then London (~9600 km). Both Seattle
        # and Sydney are closer than London — they should both be in
        # the top-3 cascade.
        assert "lhr" not in cascade

    def test_top_n_caps_at_pop_count(self) -> None:
        """Asking for more PoPs than exist returns all of them, sorted."""
        pops = (
            PoP(site_id="tk", code="tok", name="TOK", lat_deg=35.68, lon_deg=139.77),
            PoP(site_id="sea", code="sea", name="SEA", lat_deg=47.60, lon_deg=-122.33),
        )
        grid = CellGrid.from_endpoints([("tokyo_user", 35.70, 139.80)])
        table = build_cell_to_pop_nearest(grid, pops, built_at=0.0, top_n=10)
        cascade = table.pops_of(grid.cell_of("tokyo_user"))
        assert cascade == ("tok", "sea")

    def test_top_n_must_be_positive(self) -> None:
        pops = (
            PoP(site_id="tk", code="tok", name="TOK", lat_deg=35.68, lon_deg=139.77),
        )
        grid = CellGrid.from_endpoints([("tokyo_user", 35.70, 139.80)])
        with pytest.raises(ValueError, match="top_n must be positive"):
            build_cell_to_pop_nearest(grid, pops, built_at=0.0, top_n=0)

    def test_empty_pops_raises(self) -> None:
        grid = CellGrid.from_endpoints([("a", 0.0, 0.0)])
        with pytest.raises(ValueError, match="non-empty"):
            build_cell_to_pop_nearest(grid, pops=[], built_at=0.0)

    def test_version_and_built_at_recorded(self) -> None:
        pops = (
            PoP(site_id="tk", code="tok", name="TOK", lat_deg=35.68, lon_deg=139.77),
        )
        grid = CellGrid.from_endpoints([("tokyo_user", 35.70, 139.80)])
        table = build_cell_to_pop_nearest(grid, pops, built_at=999.0, version=3)
        assert table.version == 3
        assert table.built_at == 999.0


# --- build_routing_plane_nearest_pop ----------------------------------------


@pytest.mark.unit
class TestBuildRoutingPlaneNearestPoP:
    def test_plane_composes_cell_to_pop_and_sat_fibs(self) -> None:
        snap = _two_sat_snapshot()
        # Endpoint near the PoP.
        grid = CellGrid.from_endpoints([("alice", 0.0, 1.0)])
        plane = build_routing_plane_nearest_pop(snap, grid, version=5)
        # cell_to_pop has exactly one entry mapping to pop_a.
        assert plane.cell_to_pop.pop_of(grid.cell_of("alice")) == "pop_a"
        # Both sats have FIB entries for pop_a.
        assert plane.fib_of(0).route("pop_a").is_forward
        assert plane.fib_of(1).route("pop_a").is_egress
        # Version propagated through both tables.
        assert plane.version == 5
        assert plane.cell_to_pop.version == 5
        assert plane.sat_fibs[0].version == 5
        # built_at pulled from snapshot time.
        assert plane.built_at == snap.time_s


# --- NearestPoPController wiring --------------------------------------------


@pytest.mark.unit
class TestNearestPoPControllerRoutingPlane:
    def test_compute_routing_plane_end_to_end(self) -> None:
        snap = _two_sat_snapshot()
        grid = CellGrid.from_endpoints([("alice", 0.0, 1.0)])
        ctrl = NearestPoPController()
        plane = ctrl.compute_routing_plane(snap, grid, version=1)
        assert plane.fib_of(0).route("pop_a").next_hop_sat == 1
        assert plane.fib_of(1).route("pop_a").egress_gs == "gs_a"

    def test_cell_id_is_h3_int(self) -> None:
        """Sanity: plane.cell_to_pop keys are h3 int ids (not strings)."""
        snap = _two_sat_snapshot()
        grid = CellGrid.from_endpoints([("alice", 0.0, 1.0)])
        plane = NearestPoPController().compute_routing_plane(snap, grid)
        for cell_id in plane.cell_to_pop.mapping:
            assert isinstance(cell_id, int)
            # Must round-trip through the h3 string form.
            assert isinstance(cell_id_to_str(cell_id), str)
            assert latlng_to_cell_id(0.0, 1.0) == cell_id
