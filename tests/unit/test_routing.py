"""Tests for ISL routing module."""

from __future__ import annotations

import numpy as np
import pytest

from vantage.domain import ISLEdge, ISLGraph, ShellConfig
from vantage.world.satellite.routing import compute_all_pairs
from vantage.world.satellite.topology import PlusGridTopology


@pytest.fixture
def routing_graph(toy_shell: ShellConfig) -> ISLGraph:
    """Build a +Grid graph on uniform random positions for routing tests."""
    rng = np.random.default_rng(42)
    positions = np.zeros((toy_shell.total_sats, 3))
    positions[:, 0] = rng.uniform(-60, 60, toy_shell.total_sats)
    positions[:, 1] = rng.uniform(-180, 180, toy_shell.total_sats)
    positions[:, 2] = toy_shell.altitude_km
    return PlusGridTopology().build(toy_shell, positions, timeslot=0)


@pytest.mark.unit
class TestAllPairs:
    """Test all-pairs shortest path computation."""

    def test_matrix_shape(self, routing_graph: ISLGraph, toy_shell: ShellConfig) -> None:
        result = compute_all_pairs(routing_graph)
        n = toy_shell.total_sats
        assert result.delay_matrix.shape == (n, n)

    def test_diagonal_is_zero(self, routing_graph: ISLGraph, toy_shell: ShellConfig) -> None:
        result = compute_all_pairs(routing_graph)
        for i in range(toy_shell.total_sats):
            assert result.delay_matrix[i, i] == 0.0

    def test_matrix_is_symmetric(self, routing_graph: ISLGraph) -> None:
        result = compute_all_pairs(routing_graph)
        assert np.allclose(result.delay_matrix, result.delay_matrix.T)

    def test_no_unreachable_nodes(self, routing_graph: ISLGraph) -> None:
        """In a connected +Grid graph, all pairs should be reachable."""
        result = compute_all_pairs(routing_graph)
        assert not np.any(np.isinf(result.delay_matrix))

    def test_matrix_is_read_only(self, routing_graph: ISLGraph) -> None:
        """Delay matrix should be read-only to prevent accidental mutation."""
        result = compute_all_pairs(routing_graph)
        with pytest.raises(ValueError):
            result.delay_matrix[0, 0] = 999.0

    def test_metadata(self, routing_graph: ISLGraph, toy_shell: ShellConfig) -> None:
        result = compute_all_pairs(routing_graph)
        assert result.shell_id == toy_shell.shell_id
        assert result.timeslot == 0
        assert result.num_sats == toy_shell.total_sats
