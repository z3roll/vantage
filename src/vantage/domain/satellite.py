"""Satellite segment domain types.

All delay/RTT values in ms. Geographic coordinates in degrees; distances in km.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from vantage.common import access_delay


@dataclass(frozen=True, slots=True)
class ShellConfig:
    """Configuration for one orbital shell."""

    shell_id: int
    altitude_km: float
    orbit_cycle_s: float
    inclination_deg: float
    phase_shift: int
    num_orbits: int
    sats_per_orbit: int

    @property
    def total_sats(self) -> int:
        return self.num_orbits * self.sats_per_orbit

    @property
    def is_polar(self) -> bool:
        """Polar orbit: inclination in (80, 100) degrees."""
        return 80.0 < self.inclination_deg < 100.0


@dataclass(frozen=True, slots=True)
class ConstellationConfig:
    """Full constellation configuration (multiple shells)."""

    name: str
    shells: tuple[ShellConfig, ...]

    @property
    def total_sats(self) -> int:
        return sum(s.total_sats for s in self.shells)


@dataclass(frozen=True, slots=True)
class ISLEdge:
    """An inter-satellite link edge. delay is one-way propagation in ms."""

    sat_a: int
    sat_b: int
    delay: float  # one-way propagation, ms
    distance_km: float
    link_type: Literal["intra_orbit", "inter_orbit"]
    capacity_gbps: float = 20.0


@dataclass(frozen=True, slots=True)
class ISLGraph:
    """ISL topology for a single shell at a single timeslot."""

    shell_id: int
    timeslot: int
    num_sats: int
    edges: tuple[ISLEdge, ...]


@dataclass(frozen=True, slots=True)
class AccessLink:
    """Access link between a ground point and a satellite.

    sat_id is 0-based for constellation queries;
    -1 for direct-pair calls via compute_access_pair.
    delay is one-way propagation in ms.
    """

    sat_id: int
    elevation_deg: float
    slant_range_km: float
    delay: float  # one-way propagation, ms


@dataclass(frozen=True, slots=True)
class GatewayAttachments:
    """Pre-computed satellite visibility for each ground station.

    Per-GS: top-k visible satellites sorted by elevation (highest first).
    Only enabled GSs with visible satellites are included.
    """

    attachments: Mapping[str, tuple[AccessLink, ...]]  # gs_id → visible sats


@dataclass(frozen=True, slots=True)
class SatelliteState:
    """Frozen satellite segment state at a single timeslot.

    delay_matrix values are one-way ISL propagation in ms.
    Metadata (timeslot, shell_id, num_sats) derived from graph.
    """

    positions: NDArray[np.float64]  # (n_sats, 3) [lat, lon, alt], read-only
    graph: ISLGraph  # edges with delay and capacity_gbps
    delay_matrix: NDArray[np.float64]  # (n, n) one-way ISL delay in ms, read-only
    predecessor_matrix: NDArray[np.int32]  # (n, n), read-only, -1 = no path
    gateway_attachments: GatewayAttachments = field(
        default_factory=lambda: GatewayAttachments(attachments={})
    )  # per-GS visible satellites, populated by WorldModel

    @property
    def timeslot(self) -> int:
        return self.graph.timeslot

    @property
    def shell_id(self) -> int:
        return self.graph.shell_id

    @property
    def num_sats(self) -> int:
        return self.graph.num_sats

    def __post_init__(self) -> None:
        n = self.graph.num_sats
        if self.positions.shape != (n, 3):
            raise ValueError(f"positions shape {self.positions.shape} != expected ({n}, 3)")
        if self.delay_matrix.shape != (n, n):
            raise ValueError(f"delay_matrix shape {self.delay_matrix.shape} != ({n}, {n})")
        if self.predecessor_matrix.shape != (n, n):
            raise ValueError(
                f"predecessor_matrix shape {self.predecessor_matrix.shape} != ({n}, {n})"
            )

    def compute_satellite_rtt(
        self,
        user_sat: int,
        egress_sat: int,
        terminal_lat: float,
        terminal_lon: float,
        gs_lat: float,
        gs_lon: float,
    ) -> float:
        """Satellite propagation RTT = (uplink + ISL + downlink) × 2. Returns ms.

        Computes access link delays internally from positions.
        """
        uplink = access_delay(
            terminal_lat, terminal_lon,
            float(self.positions[user_sat, 0]),
            float(self.positions[user_sat, 1]),
            float(self.positions[user_sat, 2]),
        )
        isl = float(self.delay_matrix[user_sat, egress_sat])
        downlink = access_delay(
            gs_lat, gs_lon,
            float(self.positions[egress_sat, 0]),
            float(self.positions[egress_sat, 1]),
            float(self.positions[egress_sat, 2]),
        )
        return (uplink + isl + downlink) * 2
