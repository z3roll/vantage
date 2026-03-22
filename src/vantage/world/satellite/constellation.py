"""Constellation model: generates satellite positions over time.

Supports multiple generation strategies via Protocol.
Current implementation: XML-based (matching StarPerf's approach).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from math import sqrt
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from sgp4.api import WGS72, Satrec
from skyfield.api import EarthSatellite, load, wgs84

from vantage.common.constants import EARTH_RADIUS_KM, EARTH_RADIUS_M
from vantage.domain import ConstellationConfig, ShellConfig

# Orbital mechanics constants (StarPerf-specific)
GM_M3_S2 = 3.9860044e14
BSTAR = 2.8098e-05


class ConstellationModel(Protocol):
    """Protocol for constellation position generation."""

    @property
    def config(self) -> ConstellationConfig: ...

    def positions_array_at(
        self, timeslot: int, shell_id: int
    ) -> NDArray[np.float64]:
        """Return raw numpy array of positions (N, 3) for a shell at a timeslot."""
        ...

    @property
    def num_timeslots(self) -> int:
        """Total number of timeslots per orbital period."""
        ...

    def num_timeslots_for_shell(self, shell_id: int) -> int:
        """Number of timeslots for a specific shell."""
        ...


def _require_xml_field(elem: ET.Element, tag: str, shell_id: int) -> str:
    """Extract a required XML field, raising ValueError if missing."""
    text = elem.findtext(tag)
    if text is None:
        raise ValueError(
            f"Shell {shell_id}: required field <{tag}> missing in XML config"
        )
    return text


def parse_xml_config(xml_path: str | Path) -> ConstellationConfig:
    """Parse a StarPerf-format XML constellation config file.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        ValueError: If required fields are missing or have invalid values.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML config not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    num_shells_text = root.findtext("number_of_shells")
    if num_shells_text is None:
        raise ValueError("XML config missing <number_of_shells>")
    num_shells = int(num_shells_text)
    name = xml_path.stem

    shells: list[ShellConfig] = []
    for i in range(1, num_shells + 1):
        shell_elem = root.find(f"shell{i}")
        if shell_elem is None:
            raise ValueError(f"XML config missing <shell{i}> element")

        altitude = float(_require_xml_field(shell_elem, "altitude", i))
        orbit_cycle = float(_require_xml_field(shell_elem, "orbit_cycle", i))
        num_orbits = int(_require_xml_field(shell_elem, "number_of_orbit", i))
        sats_per_orbit = int(
            _require_xml_field(shell_elem, "number_of_satellite_per_orbit", i)
        )

        if altitude <= 0:
            raise ValueError(f"Shell {i}: altitude must be positive, got {altitude}")
        if orbit_cycle <= 0:
            raise ValueError(f"Shell {i}: orbit_cycle must be positive, got {orbit_cycle}")
        if num_orbits <= 0:
            raise ValueError(f"Shell {i}: number_of_orbit must be positive, got {num_orbits}")
        if sats_per_orbit <= 0:
            raise ValueError(
                f"Shell {i}: number_of_satellite_per_orbit must be positive, got {sats_per_orbit}"
            )

        shells.append(
            ShellConfig(
                shell_id=i,
                altitude_km=altitude,
                orbit_cycle_s=orbit_cycle,
                inclination_deg=float(
                    _require_xml_field(shell_elem, "inclination", i)
                ),
                phase_shift=int(shell_elem.findtext("phase_shift", default="0")),
                num_orbits=num_orbits,
                sats_per_orbit=sats_per_orbit,
            )
        )

    if not shells:
        raise ValueError("XML config contains no shells")

    return ConstellationConfig(name=name, shells=tuple(shells))


class XMLConstellationModel:
    """Constellation model using XML config + SGP4/Skyfield propagation.

    Matches StarPerf's orbit_configuration.py for bit-compatible results.
    Positions are computed for the full orbital period and cached in memory.
    """

    def __init__(
        self,
        xml_path: str | Path,
        dt_s: float = 15.0,
        epoch_year: int = 2024,
        epoch_month: int = 1,
        epoch_day: int = 1,
    ) -> None:
        if dt_s < 1.0:
            raise ValueError(f"dt_s must be >= 1.0, got {dt_s}")
        self._config = parse_xml_config(xml_path)
        self._dt_s = dt_s
        self._epoch = (epoch_year, epoch_month, epoch_day)

        # Cache: shell_id -> (num_timeslots, positions_array)
        # positions_array shape: (num_timeslots, num_sats, 3) for (lat, lon, alt)
        self._cache: dict[int, NDArray[np.float64]] = {}
        self._num_timeslots_per_shell: dict[int, int] = {}

        self._precompute_all_shells()

    @property
    def config(self) -> ConstellationConfig:
        return self._config

    @property
    def num_timeslots(self) -> int:
        """Number of timeslots for the first shell (primary orbital period)."""
        if not self._num_timeslots_per_shell:
            return 0
        first_shell_id = self._config.shells[0].shell_id
        return self._num_timeslots_per_shell[first_shell_id]

    def num_timeslots_for_shell(self, shell_id: int) -> int:
        """Number of timeslots for a specific shell."""
        return self._num_timeslots_per_shell[shell_id]

    def positions_array_at(
        self, timeslot: int, shell_id: int
    ) -> NDArray[np.float64]:
        """Return raw numpy array of positions (N, 3) for a shell at a timeslot.

        Columns: [lat_deg, lon_deg, alt_km].
        Returns a read-only view to protect cached state.
        """
        num_ts = self._num_timeslots_per_shell[shell_id]
        if timeslot < 0 or timeslot >= num_ts:
            raise ValueError(
                f"Timeslot {timeslot} out of range [0, {num_ts}) for shell {shell_id}"
            )
        view = self._cache[shell_id][timeslot]
        view.flags.writeable = False
        return view

    def _get_shell(self, shell_id: int) -> ShellConfig:
        for shell in self._config.shells:
            if shell.shell_id == shell_id:
                return shell
        raise ValueError(f"Shell {shell_id} not found in constellation config")

    def _precompute_all_shells(self) -> None:
        """Compute and cache positions for all shells."""
        ts = load.timescale()

        for shell in self._config.shells:
            num_timeslots = int(shell.orbit_cycle_s / self._dt_s)
            self._num_timeslots_per_shell[shell.shell_id] = num_timeslots

            total_sats = shell.total_sats
            # (num_timeslots, num_sats, 3) for lat/lon/alt
            positions = np.zeros((num_timeslots, total_sats, 3), dtype=np.float64)

            # Build RAAN distribution
            if shell.is_polar:
                raan_list = [i * 180.0 / shell.num_orbits for i in range(shell.num_orbits)]
            else:
                raan_list = [i * 360.0 / shell.num_orbits for i in range(shell.num_orbits)]

            # Mean anomaly base distribution (per-satellite spacing within orbit)
            ma_list = [j * 360.0 / shell.sats_per_orbit for j in range(shell.sats_per_orbit)]

            # Compute mean motion (radians/min) from altitude
            # sgp4init's no_kozai parameter expects radians per minute
            a_m = EARTH_RADIUS_M + shell.altitude_km * 1000.0
            mean_motion_rad_s = sqrt(GM_M3_S2 / (a_m**3))
            mean_motion_rad_min = mean_motion_rad_s * 60.0

            # Time array
            t_ts = ts.utc(
                self._epoch[0],
                self._epoch[1],
                self._epoch[2],
                0,
                0,
                list(range(0, int(shell.orbit_cycle_s), int(self._dt_s))),
            )

            # Generate satellites
            sat_id = 0
            earth_sats = _build_earth_satellites(
                shell, raan_list, ma_list, mean_motion_rad_min, ts
            )

            for earth_sat in earth_sats:
                geocentric = earth_sat.at(t_ts)
                subpoint = wgs84.subpoint(geocentric)

                positions[:num_timeslots, sat_id, 0] = subpoint.latitude.degrees[
                    :num_timeslots
                ]
                positions[:num_timeslots, sat_id, 1] = subpoint.longitude.degrees[
                    :num_timeslots
                ]
                positions[:num_timeslots, sat_id, 2] = subpoint.elevation.km[
                    :num_timeslots
                ]
                sat_id += 1

            self._cache[shell.shell_id] = positions


def _build_earth_satellites(
    shell: ShellConfig,
    raan_list: Sequence[float],
    ma_list: Sequence[float],
    mean_motion_rad_min: float,
    ts: object,
) -> list[EarthSatellite]:
    """Build Skyfield EarthSatellite objects for all sats in a shell.

    Matches StarPerf's SGP4 initialization parameters exactly.

    Note: StarPerf does not apply phase_shift in orbit_configuration.py
    (the field is parsed but unused). We apply it here for correctness:
    Walker-delta phase offset = phase_shift * 360 / total_sats * orbit_idx.
    When phase_shift=0, behavior is identical to StarPerf.
    """
    satellites: list[EarthSatellite] = []
    satnum = 0

    # Walker-delta phase offset per orbit (degrees)
    total_sats = shell.num_orbits * shell.sats_per_orbit
    phase_offset_per_orbit = (
        shell.phase_shift * 360.0 / total_sats if total_sats > 0 else 0.0
    )

    for orbit_idx in range(shell.num_orbits):
        raan_deg = raan_list[orbit_idx]
        orbit_ma_offset = phase_offset_per_orbit * orbit_idx
        for sat_idx in range(shell.sats_per_orbit):
            ma_deg = (ma_list[sat_idx] + orbit_ma_offset) % 360.0

            satrec = Satrec()
            satrec.sgp4init(
                WGS72,
                "i",  # improved mode
                satnum,
                25544.0,  # epoch (days since 1949-12-31, approximate)
                BSTAR,
                6.969196665e-13,  # ndot
                0.0,  # nddot
                0.0,  # eccentricity (circular)
                0.0,  # argument of perigee
                np.radians(shell.inclination_deg),
                np.radians(ma_deg),
                mean_motion_rad_min,
                np.radians(raan_deg),
            )

            earth_sat = EarthSatellite.from_satrec(satrec, ts)
            satellites.append(earth_sat)
            satnum += 1

    return satellites
