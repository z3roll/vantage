"""Shared test fixtures for Vantage simulator."""

from pathlib import Path

import pytest

from vantage.domain import ConstellationConfig, ShellConfig


STARPERF_XML_DIR = Path("/Users/zerol/PhD/starperf/config/XML_constellation")


@pytest.fixture
def starlink_xml_path() -> Path:
    """Path to StarPerf's Starlink XML config."""
    return STARPERF_XML_DIR / "Starlink.xml"


@pytest.fixture
def toy_shell() -> ShellConfig:
    """Small shell config for fast tests (72 sats)."""
    return ShellConfig(
        shell_id=1,
        altitude_km=550.0,
        orbit_cycle_s=5731.0,
        inclination_deg=53.0,
        phase_shift=1,
        num_orbits=6,
        sats_per_orbit=12,
    )


@pytest.fixture
def toy_constellation(toy_shell: ShellConfig) -> ConstellationConfig:
    """Small constellation config for fast tests."""
    return ConstellationConfig(name="toy", shells=(toy_shell,))


@pytest.fixture
def polar_shell() -> ShellConfig:
    """Polar orbit shell (97.5° inclination)."""
    return ShellConfig(
        shell_id=6,
        altitude_km=552.0,
        orbit_cycle_s=5733.0,
        inclination_deg=97.5,
        phase_shift=1,
        num_orbits=6,
        sats_per_orbit=58,
    )
