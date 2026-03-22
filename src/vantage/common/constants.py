"""Physical constants for the Vantage TE system."""

EARTH_RADIUS_KM = 6371.0
"""Mean Earth radius (km)."""

EARTH_RADIUS_M = 6_371_393.0
"""Mean Earth radius (m). Used by constellation orbit calculations."""

C_VACUUM_KM_S = 300_000.0
"""Speed of light in vacuum (km/s)."""

C_FIBER_KM_S = 200_000.0
"""Effective speed of light in optical fiber (km/s). ~2/3 × c_vacuum."""

DEFAULT_DETOUR_FACTOR = 1.5
"""Default fiber routing detour factor (actual path / great-circle distance)."""

DEFAULT_MIN_ELEVATION_DEG = 25.0
"""Default minimum satellite elevation angle for user access links (degrees)."""
