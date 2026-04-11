"""Tests for predecessor matrix and SatelliteSegment facade."""

from __future__ import annotations

import numpy as np
import pytest

from vantage.domain import ISLEdge, ISLGraph
from vantage.world.satellite import SatelliteSegment
from vantage.world.satellite.routing import compute_all_pairs
from vantage.world.satellite.topology import PlusGridTopology


# ---------------------------------------------------------------------------
# Predecessor matrix
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPredecessorMatrix:
    """Test predecessor matrix generation."""

    @pytest.fixture
    def linear_graph(self) -> ISLGraph:
        """0 -- 1 -- 2 -- 3 (chain)."""
        edges = tuple(
            ISLEdge(sat_a=i, sat_b=i + 1, delay=1.0 * (i + 1),
                    distance_km=300.0, link_type="intra_orbit")
            for i in range(3)
        )
        return ISLGraph(shell_id=1, timeslot=0, num_sats=4, edges=edges)

    @pytest.fixture
    def triangle_graph(self) -> ISLGraph:
        """0 -- 1 (fast), 1 -- 2 (fast), 0 -- 2 (slow)."""
        edges = (
            ISLEdge(0, 1, delay=1.0, distance_km=300.0, link_type="intra_orbit"),
            ISLEdge(1, 2, delay=1.0, distance_km=300.0, link_type="intra_orbit"),
            ISLEdge(0, 2, delay=10.0, distance_km=3000.0, link_type="inter_orbit"),
        )
        return ISLGraph(shell_id=1, timeslot=0, num_sats=3, edges=edges)

    def test_predecessor_matrix_shape(self, linear_graph: ISLGraph) -> None:
        result = compute_all_pairs(linear_graph)
        assert result.predecessor_matrix.shape == (4, 4)
        assert result.predecessor_matrix.dtype == np.int32

    def test_predecessor_matrix_read_only(self, linear_graph: ISLGraph) -> None:
        result = compute_all_pairs(linear_graph)
        assert not result.predecessor_matrix.flags.writeable

    def test_predecessor_diagonal_is_self(self, linear_graph: ISLGraph) -> None:
        result = compute_all_pairs(linear_graph)
        for i in range(4):
            assert result.predecessor_matrix[i, i] == i

    def test_predecessor_chooses_shortest(self, triangle_graph: ISLGraph) -> None:
        """0->2: direct edge is 10.0, via 1 is 2.0. Should choose via 1."""
        result = compute_all_pairs(triangle_graph)
        # predecessor_matrix[0, 2] should be 1 (came via node 1)
        assert result.predecessor_matrix[0, 2] == 1

    def test_delay_matrix_linear_path(self, linear_graph: ISLGraph) -> None:
        """Delay from 0 to 3 should be sum of edge delays: 1.0 + 2.0 + 3.0 = 6.0."""
        result = compute_all_pairs(linear_graph)
        assert abs(result.delay_matrix[0, 3] - 6.0) < 1e-12

    def test_disconnected_subgraph_consistency(self) -> None:
        """Nodes 0-1-2 connected chain; node 3 isolated."""
        edges = (
            ISLEdge(0, 1, 1.0, 300.0, "intra_orbit"),
            ISLEdge(1, 2, 1.0, 300.0, "intra_orbit"),
        )
        graph = ISLGraph(shell_id=1, timeslot=0, num_sats=4, edges=edges)
        result = compute_all_pairs(graph)
        # Unreachable: delay inf, predecessor -1
        assert result.delay_matrix[0, 3] == np.inf
        assert result.predecessor_matrix[0, 3] == -1
        # Reachable: delay finite, predecessor != -1
        assert result.delay_matrix[0, 2] < np.inf
        assert result.predecessor_matrix[0, 2] != -1


# ---------------------------------------------------------------------------
# ISLEdge capacity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestISLEdgeCapacity:
    """Test ISLEdge capacity_gbps field."""

    def test_default_capacity(self) -> None:
        edge = ISLEdge(0, 1, 1.0, 300.0, "intra_orbit")
        # Default matches Starlink v2 mini laser ISL.
        assert edge.capacity_gbps == 96.0

    def test_custom_capacity(self) -> None:
        edge = ISLEdge(0, 1, 1.0, 300.0, "intra_orbit", capacity_gbps=40.0)
        assert edge.capacity_gbps == 40.0


# ---------------------------------------------------------------------------
# SatelliteSegment facade
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSatelliteSegment:
    """Test SatelliteSegment facade with real constellation data."""

    @pytest.fixture
    def segment(self, starlink_xml_path: str) -> SatelliteSegment:
        from vantage.world.satellite.constellation import XMLConstellationModel

        constellation = XMLConstellationModel(starlink_xml_path, dt_s=15.0)
        return SatelliteSegment(
            constellation=constellation,
            topology_builder=PlusGridTopology(),
            shell_id=1,
        )

    def test_state_at_returns_satellite_state(
        self, segment: SatelliteSegment
    ) -> None:
        state = segment.state_at(0)
        assert state.timeslot == 0
        assert state.shell_id == 1
        assert state.num_sats == segment.num_sats

    def test_state_positions_shape(self, segment: SatelliteSegment) -> None:
        state = segment.state_at(0)
        assert state.positions.shape == (segment.num_sats, 3)
        assert not state.positions.flags.writeable

    def test_state_delay_matrix_shape(self, segment: SatelliteSegment) -> None:
        state = segment.state_at(0)
        n = segment.num_sats
        assert state.delay_matrix.shape == (n, n)
        assert not state.delay_matrix.flags.writeable

    def test_state_predecessor_matrix_shape(
        self, segment: SatelliteSegment
    ) -> None:
        state = segment.state_at(0)
        n = segment.num_sats
        assert state.predecessor_matrix.shape == (n, n)
        assert state.predecessor_matrix.dtype == np.int32

    def test_state_graph_edge_count(self, segment: SatelliteSegment) -> None:
        state = segment.state_at(0)
        assert len(state.graph.edges) > 0

    def test_shell_id_internalized(self, segment: SatelliteSegment) -> None:
        """shell_id is not in state_at() signature."""
        assert segment.shell_id == 1
        state = segment.state_at(0)
        assert state.shell_id == 1
