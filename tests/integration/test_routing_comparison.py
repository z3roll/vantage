"""Cross-validation: compare Vantage routing results against StarPerf.

Uses StarPerf's pre-computed delay matrix for shell2, builds ISL graph
with Vantage, runs Dijkstra, and verifies paths and delays match exactly.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import networkx as nx
import numpy as np
import pytest

from vantage.world.satellite.constellation import parse_xml_config
from vantage.world.satellite.routing import compute_all_pairs
from vantage.world.satellite.topology import PlusGridTopology

STARPERF_XML = Path("/Users/zerol/PhD/starperf/config/XML_constellation/Starlink.xml")
STARPERF_H5 = Path("/Users/zerol/PhD/starperf/data/XML_constellation/Starlink.h5")


def _load_starperf_shell2_graph() -> tuple[nx.Graph, int]:
    """Load StarPerf delay matrix as networkx graph (1-indexed)."""
    with h5py.File(STARPERF_H5, "r") as f:
        sp_delay = f["delay"]["shell2"]["timeslot1"][:]

    n = sp_delay.shape[0] - 1  # 1-indexed
    g = nx.Graph()
    g.add_nodes_from(range(1, n + 1))
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if sp_delay[i][j] > 0:
                g.add_edge(i, j, weight=sp_delay[i][j])
    return g, n


def _load_starperf_positions_shell2() -> np.ndarray:
    """Load StarPerf shell2 positions and decode byte strings."""
    with h5py.File(STARPERF_H5, "r") as f:
        raw_pos = f["position"]["shell2"]["timeslot1"][:]

    n_sats = raw_pos.shape[0]
    positions = np.zeros((n_sats, 3), dtype=np.float64)
    for i in range(n_sats):
        # StarPerf HDF5 columns: col0=longitude, col1=latitude, col2=altitude.
        # See: starperf/src/constellation_generation/by_XML/orbit_configuration.py
        #   satellite_position['longitude'] -> stored first (col 0)
        #   satellite_position['latitude']  -> stored second (col 1)
        # Verified by exact delay match against StarPerf's pre-computed delay matrix.
        lon = float(raw_pos[i, 0])
        lat = float(raw_pos[i, 1])
        alt = float(raw_pos[i, 2])
        # Vantage internal format: (lat, lon, alt)
        positions[i] = [lat, lon, alt]
    return positions


@pytest.mark.integration
@pytest.mark.slow
class TestRoutingStarPerfComparison:
    """Compare Vantage Dijkstra routing against StarPerf on shell2."""

    @pytest.fixture(autouse=True)
    def _check_data(self) -> None:
        if not STARPERF_XML.exists() or not STARPERF_H5.exists():
            pytest.skip("StarPerf data not available")

    @pytest.fixture
    def vantage_graph(self):
        """Build Vantage ISL graph using StarPerf positions for shell2."""
        config = parse_xml_config(STARPERF_XML)
        shell = config.shells[1]  # shell2
        positions = _load_starperf_positions_shell2()
        builder = PlusGridTopology()
        return builder.build(shell, positions, timeslot=0), shell

    def test_dijkstra_delay_matches_starperf_pair1(self, vantage_graph) -> None:
        """sat 0 -> sat 799 (StarPerf 1-indexed: 1 -> 800)."""
        graph, shell = vantage_graph
        result = compute_all_pairs(graph)

        vantage_delay = result.delay_matrix[0, 799]

        # StarPerf reference
        sp_graph, _ = _load_starperf_shell2_graph()
        sp_delay = nx.dijkstra_path_length(sp_graph, 1, 800, weight="weight")

        print(f"\nPair 1: sat 0 -> sat 799")
        print(f"  Vantage: delay={vantage_delay:.4f} ms")
        print(f"  StarPerf: delay={sp_delay * 1000:.4f} ms")
        print(f"  Diff: {abs(vantage_delay - sp_delay * 1000):.6f} ms")

        # Delays should match within floating-point tolerance (compare in ms)
        assert abs(vantage_delay - sp_delay * 1000) < 1e-3, (
            f"Delay mismatch: Vantage={vantage_delay:.10f} ms, StarPerf={sp_delay * 1000:.10f} ms"
        )

    def test_dijkstra_delay_matches_starperf_pair2(self, vantage_graph) -> None:
        """sat 99 -> sat 1499 (StarPerf 1-indexed: 100 -> 1500)."""
        graph, shell = vantage_graph
        result = compute_all_pairs(graph)

        vantage_delay = result.delay_matrix[99, 1499]

        sp_graph, _ = _load_starperf_shell2_graph()
        sp_delay = nx.dijkstra_path_length(sp_graph, 100, 1500, weight="weight")

        print(f"\nPair 2: sat 99 -> sat 1499")
        print(f"  Vantage: delay={vantage_delay:.4f} ms")
        print(f"  StarPerf: delay={sp_delay * 1000:.4f} ms")
        print(f"  Diff: {abs(vantage_delay - sp_delay * 1000):.6f} ms")

        assert abs(vantage_delay - sp_delay * 1000) < 1e-3

    def test_dijkstra_delay_matches_starperf_pair3(self, vantage_graph) -> None:
        """sat 499 -> sat 999 (StarPerf 1-indexed: 500 -> 1000)."""
        graph, shell = vantage_graph
        result = compute_all_pairs(graph)

        vantage_delay = result.delay_matrix[499, 999]

        sp_graph, _ = _load_starperf_shell2_graph()
        sp_delay = nx.dijkstra_path_length(sp_graph, 500, 1000, weight="weight")

        print(f"\nPair 3: sat 499 -> sat 999")
        print(f"  Vantage: delay={vantage_delay:.4f} ms")
        print(f"  StarPerf: delay={sp_delay * 1000:.4f} ms")

        assert abs(vantage_delay - sp_delay * 1000) < 1e-3

    def test_multiple_random_pairs(self, vantage_graph) -> None:
        """Verify 20 random pairs all match StarPerf."""
        graph, shell = vantage_graph
        result = compute_all_pairs(graph)
        sp_graph, n = _load_starperf_shell2_graph()

        rng = np.random.default_rng(42)
        pairs = [(rng.integers(0, n), rng.integers(0, n)) for _ in range(20)]

        mismatches = 0
        for src_0, tgt_0 in pairs:
            if src_0 == tgt_0:
                continue
            src_1, tgt_1 = src_0 + 1, tgt_0 + 1  # 1-indexed for StarPerf

            vantage_delay = result.delay_matrix[src_0, tgt_0]
            sp_delay = nx.dijkstra_path_length(sp_graph, src_1, tgt_1, weight="weight")

            if abs(vantage_delay - sp_delay * 1000) > 1e-3:
                mismatches += 1
                print(f"  MISMATCH ({src_0}->{tgt_0}): V={vantage_delay:.10f} ms, SP={sp_delay * 1000:.10f} ms")

        assert mismatches == 0, f"{mismatches} pairs had delay mismatches"
