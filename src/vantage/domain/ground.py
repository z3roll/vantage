"""Ground infrastructure domain types.

All delay values in ms. Geographic coordinates in degrees; distances in km.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroundStation:
    """A Starlink ground station (gateway)."""

    gs_id: str
    lat_deg: float
    lon_deg: float
    country: str
    town: str
    num_antennas: int
    min_elevation_deg: float
    enabled: bool
    uplink_ghz: float
    downlink_ghz: float
    min_capacity: float
    max_capacity: float
    temporary: bool


@dataclass(frozen=True, slots=True)
class PoP:
    """A Starlink Point of Presence."""

    site_id: str
    code: str
    name: str
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True, slots=True)
class GSPoPEdge:
    """Static fiber connection between a ground station and a PoP."""

    gs_id: str
    pop_code: str
    distance_km: float
    backhaul_delay: float  # one-way propagation, ms
    capacity_gbps: float
