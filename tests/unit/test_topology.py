"""Tests for ISL topology builder."""

from __future__ import annotations

import numpy as np
import pytest

from vantage.domain import ShellConfig
from vantage.world.satellite.topology import (
    C_VACUUM_KM_S,
    PlusGridTopology,
    _haversine_distance_km,
    build_adjacency,
)


@pytest.mark.unit
class TestHaversineDistance:
    """Test haversine distance computation."""

    def test_same_point_returns_zero(self) -> None:
        pos = np.array([0.0, 0.0, 550.0])
        assert _haversine_distance_km(pos, pos) == 0.0

    def test_equator_quarter_circle(self) -> None:
        """Two sats on equator, 90° apart at 550km altitude."""
        pos_a = np.array([0.0, 0.0, 550.0])
        pos_b = np.array([0.0, 90.0, 550.0])
        dist = _haversine_distance_km(pos_a, pos_b)
        # Expected: π/2 * (6371 + 550) ≈ 10862 km
        expected = np.pi / 2 * (6371.0 + 550.0)
        assert abs(dist - expected) < 1.0  # within 1 km

    def test_uses_average_altitude(self) -> None:
        """Distance should use average of both altitudes."""
        pos_a = np.array([0.0, 0.0, 500.0])
        pos_b = np.array([0.0, 1.0, 600.0])
        dist = _haversine_distance_km(pos_a, pos_b)
        # Average alt = 550, effective radius = 6921
        effective_r = 6371.0 + 550.0
        expected = np.radians(1.0) * effective_r
        assert abs(dist - expected) < 0.1

    def test_symmetric(self) -> None:
        pos_a = np.array([30.0, 45.0, 550.0])
        pos_b = np.array([-20.0, 120.0, 548.0])
        assert _haversine_distance_km(pos_a, pos_b) == _haversine_distance_km(pos_b, pos_a)


@pytest.mark.unit
class TestPlusGridTopology:
    """Test +Grid ISL topology construction."""

    def _make_uniform_positions(self, shell: ShellConfig) -> np.ndarray:
        """Create fake uniform positions for testing topology structure."""
        n = shell.total_sats
        rng = np.random.default_rng(42)
        positions = np.zeros((n, 3))
        positions[:, 0] = rng.uniform(-60, 60, n)  # lat
        positions[:, 1] = rng.uniform(-180, 180, n)  # lon
        positions[:, 2] = shell.altitude_km  # alt
        return positions

    def test_edge_count_non_polar(self, toy_shell: ShellConfig) -> None:
        """Non-polar: 2 edge types, each sat has 4 links (stored undirected)."""
        builder = PlusGridTopology()
        positions = self._make_uniform_positions(toy_shell)
        graph = builder.build(toy_shell, positions, timeslot=0)

        n_orbits = toy_shell.num_orbits
        n_sats = toy_shell.sats_per_orbit

        # Intra-orbit edges: n_orbits * n_sats (each orbit is a ring)
        expected_intra = n_orbits * n_sats
        # Inter-orbit edges: n_orbits * n_sats (including wrap-around for non-polar)
        expected_inter = n_orbits * n_sats
        expected_total = expected_intra + expected_inter

        assert len(graph.edges) == expected_total

    def test_no_wrap_inter_orbit_for_polar(self, polar_shell: ShellConfig) -> None:
        """Polar orbits should NOT have inter-orbit wrap-around ISLs."""
        builder = PlusGridTopology()
        positions = self._make_uniform_positions(polar_shell)
        graph = builder.build(polar_shell, positions, timeslot=0)

        n_orbits = polar_shell.num_orbits
        n_sats = polar_shell.sats_per_orbit

        # Intra-orbit: n_orbits * n_sats
        expected_intra = n_orbits * n_sats
        # Inter-orbit: (n_orbits - 1) * n_sats (no wrap-around!)
        expected_inter = (n_orbits - 1) * n_sats
        expected_total = expected_intra + expected_inter

        assert len(graph.edges) == expected_total

    def test_all_delays_positive(self, toy_shell: ShellConfig) -> None:
        builder = PlusGridTopology()
        positions = self._make_uniform_positions(toy_shell)
        graph = builder.build(toy_shell, positions, timeslot=0)

        for edge in graph.edges:
            assert edge.delay > 0
            assert edge.distance_km > 0

    def test_delay_equals_distance_over_c(self, toy_shell: ShellConfig) -> None:
        builder = PlusGridTopology()
        positions = self._make_uniform_positions(toy_shell)
        graph = builder.build(toy_shell, positions, timeslot=0)

        for edge in graph.edges:
            expected_delay = edge.distance_km / C_VACUUM_KM_S * 1000  # ms
            assert abs(edge.delay - expected_delay) < 1e-9

    def test_no_self_loops(self, toy_shell: ShellConfig) -> None:
        builder = PlusGridTopology()
        positions = self._make_uniform_positions(toy_shell)
        graph = builder.build(toy_shell, positions, timeslot=0)

        for edge in graph.edges:
            assert edge.sat_a != edge.sat_b

    def test_no_duplicate_edges(self, toy_shell: ShellConfig) -> None:
        builder = PlusGridTopology()
        positions = self._make_uniform_positions(toy_shell)
        graph = builder.build(toy_shell, positions, timeslot=0)

        edge_set = {(e.sat_a, e.sat_b) for e in graph.edges}
        assert len(edge_set) == len(graph.edges)

    def test_graph_metadata(self, toy_shell: ShellConfig) -> None:
        builder = PlusGridTopology()
        positions = self._make_uniform_positions(toy_shell)
        graph = builder.build(toy_shell, positions, timeslot=5)

        assert graph.shell_id == toy_shell.shell_id
        assert graph.timeslot == 5
        assert graph.num_sats == toy_shell.total_sats


@pytest.mark.unit
class TestBuildAdjacency:
    """Test adjacency list construction."""

    def test_adjacency_is_symmetric(self, toy_shell: ShellConfig) -> None:
        builder = PlusGridTopology()
        rng = np.random.default_rng(42)
        positions = np.zeros((toy_shell.total_sats, 3))
        positions[:, 0] = rng.uniform(-60, 60, toy_shell.total_sats)
        positions[:, 1] = rng.uniform(-180, 180, toy_shell.total_sats)
        positions[:, 2] = toy_shell.altitude_km
        graph = builder.build(toy_shell, positions, timeslot=0)
        adj = build_adjacency(graph)

        for node, neighbors in adj.items():
            for neighbor, delay in neighbors:
                # neighbor should also have node in its adjacency
                neighbor_nodes = {n for n, _ in adj[neighbor]}
                assert node in neighbor_nodes

    def test_non_polar_all_sats_have_4_neighbors(self, toy_shell: ShellConfig) -> None:
        builder = PlusGridTopology()
        rng = np.random.default_rng(42)
        positions = np.zeros((toy_shell.total_sats, 3))
        positions[:, 0] = rng.uniform(-60, 60, toy_shell.total_sats)
        positions[:, 1] = rng.uniform(-180, 180, toy_shell.total_sats)
        positions[:, 2] = toy_shell.altitude_km
        graph = builder.build(toy_shell, positions, timeslot=0)
        adj = build_adjacency(graph)

        for sat_id in range(toy_shell.total_sats):
            assert len(adj[sat_id]) == 4, f"Satellite {sat_id} has {len(adj[sat_id])} neighbors"
