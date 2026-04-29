"""Time-Varying ISL Graph: fixed topology + time-varying edge weights.

The +Grid adjacency structure is built once at construction. Each
timeslot, only edge weights (propagation delays) are recomputed from
satellite positions. Uses scipy sparse Dijkstra (C implementation)
for all-pairs shortest path — 10-50x faster than networkx.

This is the core optimization for the satellite routing pipeline:
the ISL topology never changes, only distances do as satellites orbit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from vantage.common.constants import C_VACUUM_KM_S, EARTH_RADIUS_KM
from vantage.model.satellite.state import ISLEdge, ISLGraph, ShellConfig
from vantage.model.satellite.routing import RoutingResult


class TimeVaryingISLGraph:
    """Fixed-topology ISL graph with time-varying edge weights.

    Construction builds the +Grid adjacency once. ``compute_state()``
    takes satellite positions, vectorizes edge weight computation,
    and runs scipy sparse Dijkstra — all in C, no Python loops.
    """

    def __init__(self, shell: ShellConfig) -> None:
        self._shell = shell
        self._n = shell.total_sats

        # Build fixed edge structure (src, dst pairs)
        src_list: list[int] = []
        dst_list: list[int] = []
        type_list: list[str] = []
        self._build_structure(shell, src_list, dst_list, type_list)

        self._src = np.array(src_list, dtype=np.int32)
        self._dst = np.array(dst_list, dtype=np.int32)
        self._types = tuple(type_list)
        self._num_edges = len(src_list)

    @staticmethod
    def _build_structure(
        shell: ShellConfig,
        src_list: list[int],
        dst_list: list[int],
        type_list: list[str],
    ) -> None:
        """Build +Grid adjacency (same logic as PlusGridTopology)."""
        n_orbits = shell.num_orbits
        n_sats = shell.sats_per_orbit

        for orbit_idx in range(n_orbits):
            for sat_idx in range(n_sats):
                cur_id = orbit_idx * n_sats + sat_idx

                # Intra-orbit: next satellite in same orbit
                next_sat_idx = (sat_idx + 1) % n_sats
                next_id = orbit_idx * n_sats + next_sat_idx
                src_list.append(cur_id)
                dst_list.append(next_id)
                type_list.append("intra_orbit")

                # Inter-orbit: same sat_idx in adjacent orbit
                next_orbit_idx = orbit_idx + 1
                if next_orbit_idx < n_orbits:
                    right_id = next_orbit_idx * n_sats + sat_idx
                elif next_orbit_idx == n_orbits:
                    if shell.is_polar:
                        continue
                    right_id = sat_idx
                else:
                    continue

                src_list.append(cur_id)
                dst_list.append(right_id)
                type_list.append("inter_orbit")

    def compute_state(
        self,
        positions: NDArray[np.float64],
        timeslot: int,
    ) -> tuple[ISLGraph, RoutingResult]:
        """Compute edge weights from positions and run all-pairs shortest path.

        Returns both the ISLGraph (for SatelliteState) and the
        RoutingResult (delay + predecessor matrices).
        """
        # Validate input shape
        expected_shape = (self._n, 3)
        if positions.shape != expected_shape:
            raise ValueError(
                f"positions shape {positions.shape} != expected {expected_shape}"
            )

        # Vectorized weight computation — no Python loop
        weights = self._compute_weights(positions)

        # Build scipy sparse matrix (undirected: add both directions)
        n = self._n
        row = np.concatenate([self._src, self._dst])
        col = np.concatenate([self._dst, self._src])
        data = np.concatenate([weights, weights])

        sparse = csr_matrix((data, (row, col)), shape=(n, n))

        # Scipy all-pairs Dijkstra (C implementation)
        delay_matrix, pred_matrix = shortest_path(
            sparse, directed=False, return_predecessors=True, method="D"
        )

        delay_matrix = np.ascontiguousarray(delay_matrix, dtype=np.float64)
        pred_matrix = np.ascontiguousarray(pred_matrix, dtype=np.int32)

        # Fix predecessor matrix to match legacy contract
        pred_matrix[pred_matrix == -9999] = -1
        np.fill_diagonal(pred_matrix, np.arange(n, dtype=np.int32))

        delay_matrix.flags.writeable = False
        pred_matrix.flags.writeable = False

        # Build ISLGraph for SatelliteState compatibility
        distances = weights * C_VACUUM_KM_S / 1000.0  # ms → km
        distances_rounded = np.round(distances, 3)
        delays_rounded = distances_rounded / C_VACUUM_KM_S * 1000.0

        edges = tuple(
            ISLEdge(
                sat_a=int(self._src[i]),
                sat_b=int(self._dst[i]),
                delay=float(delays_rounded[i]),
                distance_km=float(distances_rounded[i]),
                link_type=self._types[i],
            )
            for i in range(self._num_edges)
        )
        graph = ISLGraph(
            shell_id=self._shell.shell_id,
            timeslot=timeslot,
            num_sats=n,
            edges=edges,
        )

        routing = RoutingResult(
            shell_id=self._shell.shell_id,
            timeslot=timeslot,
            num_sats=n,
            delay_matrix=delay_matrix,
            predecessor_matrix=pred_matrix,
        )

        return graph, routing

    def _compute_weights(
        self, positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Vectorized edge weight computation: haversine distance → delay (ms)."""
        src_pos = positions[self._src]  # (E, 3)
        dst_pos = positions[self._dst]  # (E, 3)

        lat1 = np.radians(src_pos[:, 0])
        lon1 = np.radians(src_pos[:, 1])
        lat2 = np.radians(dst_pos[:, 0])
        lon2 = np.radians(dst_pos[:, 1])
        avg_alt = (src_pos[:, 2] + dst_pos[:, 2]) / 2.0

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

        effective_radius = EARTH_RADIUS_KM + avg_alt
        dist_km = 2.0 * np.arcsin(np.sqrt(a)) * effective_radius

        return dist_km / C_VACUUM_KM_S * 1000  # delay in ms
