"""Ground infrastructure domain types.

All delay values in ms. Geographic coordinates in degrees; distances in km.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroundStation:
    """A Starlink ground station (gateway).

    ``max_capacity`` is the aggregate per-GS feeder throughput in
    **Gbps**: the total traffic this station can absorb across all of
    its Ka antennas from every satellite it is simultaneously serving.
    A typical 9-antenna site handles roughly ``9 × 10 = 90 Gbps``, so
    the preprocessor derives the value as ``num_antennas × 10.0``.
    This field is the authoritative per-GS feeder cap used by
    :class:`vantage.domain.capacity_view.CapacityView`.
    """

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
    max_capacity: float  # per-GS aggregate feeder Gbps
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
