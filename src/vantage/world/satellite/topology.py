"""ISL topology builder: constructs inter-satellite link graphs.

Supports multiple topology strategies via Protocol.
Current implementation: +Grid (matching StarPerf).
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from vantage.common.constants import C_VACUUM_KM_S, EARTH_RADIUS_KM
from vantage.domain import ISLEdge, ISLGraph, ShellConfig


class TopologyBuilder(Protocol):
    """Protocol for ISL topology construction."""

    def build(
        self,
        shell: ShellConfig,
        positions: NDArray[np.float64],
        timeslot: int,
    ) -> ISLGraph:
        """Build ISL graph for a shell at a timeslot.

        Args:
            shell: Shell configuration.
            positions: Array of shape (num_sats, 3) with [lat_deg, lon_deg, alt_km].
            timeslot: Current timeslot index.

        Returns:
            ISLGraph with computed edges.
        """
        ...


class PlusGridTopology:
    """+Grid ISL topology: 4 links per satellite (2 intra-orbit, 2 inter-orbit).

    Matches StarPerf's positive_Grid.py implementation exactly:
    - Intra-orbit links: each satellite connects to its neighbors within the same orbit
    - Inter-orbit links: each satellite connects to the corresponding satellite
      in adjacent orbital planes
    - Polar orbits (80°-100° inclination): no inter-orbit ISL at the last-to-first
      plane boundary
    - Distance: haversine with average satellite altitude
    - Delay: distance_km / c_vacuum, stored in ms
    """

    def build(
        self,
        shell: ShellConfig,
        positions: NDArray[np.float64],
        timeslot: int,
    ) -> ISLGraph:
        """Build +Grid ISL graph for a shell."""
        expected_shape = (shell.total_sats, 3)
        if positions.shape != expected_shape:
            raise ValueError(
                f"positions shape {positions.shape} does not match "
                f"expected {expected_shape} for shell {shell.shell_id}"
            )
        edges: list[ISLEdge] = []
        n_orbits = shell.num_orbits
        n_sats = shell.sats_per_orbit

        for orbit_idx in range(n_orbits):
            for sat_idx in range(n_sats):
                cur_id = orbit_idx * n_sats + sat_idx

                # --- Intra-orbit link (to next satellite in same orbit) ---
                next_sat_idx = (sat_idx + 1) % n_sats
                next_id = orbit_idx * n_sats + next_sat_idx

                dist = _haversine_distance_km(positions[cur_id], positions[next_id])
                delay = dist / C_VACUUM_KM_S * 1000  # ms
                edges.append(
                    ISLEdge(
                        sat_a=cur_id,
                        sat_b=next_id,
                        delay=delay,
                        distance_km=dist,
                        link_type="intra_orbit",
                    )
                )

                # --- Inter-orbit link (to same sat_idx in next orbit) ---
                next_orbit_idx = orbit_idx + 1
                if next_orbit_idx < n_orbits:
                    right_id = next_orbit_idx * n_sats + sat_idx
                elif next_orbit_idx == n_orbits:
                    # Last plane wraps to first plane
                    if shell.is_polar:
                        # Polar orbits: NO wrap-around ISL at boundary
                        continue
                    right_id = sat_idx  # wrap to orbit 0
                else:
                    continue

                dist = _haversine_distance_km(positions[cur_id], positions[right_id])
                delay = dist / C_VACUUM_KM_S * 1000  # ms
                edges.append(
                    ISLEdge(
                        sat_a=cur_id,
                        sat_b=right_id,
                        delay=delay,
                        distance_km=dist,
                        link_type="inter_orbit",
                    )
                )

        return ISLGraph(
            shell_id=shell.shell_id,
            timeslot=timeslot,
            num_sats=shell.total_sats,
            edges=tuple(edges),
        )


def _haversine_distance_km(
    pos_a: NDArray[np.float64],
    pos_b: NDArray[np.float64],
) -> float:
    """Compute haversine distance between two satellites.

    Uses average altitude as effective radius, matching StarPerf exactly.

    Args:
        pos_a: [lat_deg, lon_deg, alt_km] for satellite A.
        pos_b: [lat_deg, lon_deg, alt_km] for satellite B.

    Returns:
        Distance in km.
    """
    lat1 = radians(pos_a[0])
    lon1 = radians(pos_a[1])
    lat2 = radians(pos_b[0])
    lon2 = radians(pos_b[1])

    avg_alt_km = (pos_a[2] + pos_b[2]) / 2.0

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2

    # Effective radius = Earth radius + average altitude
    effective_radius_km = EARTH_RADIUS_KM + avg_alt_km

    distance_km = 2.0 * asin(sqrt(a)) * effective_radius_km
    return round(distance_km, 3)


def build_adjacency(graph: ISLGraph) -> dict[int, list[tuple[int, float]]]:
    """Build adjacency list from ISLGraph for shortest-path algorithms.

    Returns:
        Dict mapping sat_id -> list of (neighbor_id, delay_ms).
    """
    adj: dict[int, list[tuple[int, float]]] = {}
    for edge in graph.edges:
        adj.setdefault(edge.sat_a, []).append((edge.sat_b, edge.delay))
        adj.setdefault(edge.sat_b, []).append((edge.sat_a, edge.delay))
    return adj
