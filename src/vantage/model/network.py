"""Network snapshot and world model composition."""

from __future__ import annotations

from dataclasses import dataclass

from vantage.model.ground.infrastructure import GroundInfrastructure, GroundStation
from vantage.model.satellite.segment import SatelliteSegment
from vantage.model.satellite.state import SatelliteState, ShellConfig


@dataclass(frozen=True, slots=True)
class NetworkSnapshot:
    """Complete physical network state at simulation time ``time_s``."""

    epoch: int
    time_s: float
    satellite: SatelliteState
    infra: GroundInfrastructure


class WorldModel:
    """Composes satellite and ground segments into network snapshots."""

    def __init__(self, satellite: SatelliteSegment, ground: GroundInfrastructure) -> None:
        self._satellite = satellite
        self._ground = ground

    @property
    def shell(self) -> ShellConfig:
        return self._satellite.shell

    @property
    def ground_stations(self) -> tuple[GroundStation, ...]:
        return self._ground.ground_stations

    def snapshot_at(self, epoch: int, time_s: float) -> NetworkSnapshot:
        timeslot = self._satellite.time_to_timeslot(time_s)
        return NetworkSnapshot(
            epoch=epoch,
            time_s=time_s,
            satellite=self._satellite.state_at(timeslot),
            infra=self._ground,
        )
