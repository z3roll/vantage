"""WorldModel: composes satellite + ground into NetworkSnapshot.

This is the single entry point for the TE engine to query
physical network state. Only builds snapshots — no routing policy.
"""

from __future__ import annotations

from vantage.domain import (
    GroundStation,
    InfrastructureView,
    NetworkSnapshot,
    ShellConfig,
)
from vantage.world.ground import GroundInfrastructure
from vantage.world.satellite import SatelliteSegment


class WorldModel:
    """Composes satellite and ground segments into a complete world model.

    Produces frozen NetworkSnapshot at each epoch.
    """

    def __init__(
        self,
        satellite: SatelliteSegment,
        ground: GroundInfrastructure,
    ) -> None:
        self._satellite = satellite
        self._ground = ground

        # Pre-build frozen infrastructure view (static, never changes)
        self._infra_view = InfrastructureView(
            pops=ground.pops,
            ground_stations=ground.ground_stations,
            gs_pop_edges=ground.gs_pop_edges,
        )

    @property
    def shell(self) -> ShellConfig:
        """Return the satellite segment's :class:`ShellConfig`.

        Exposed so capacity views (and future multi-shell aware
        consumers) don't have to reach into private state.
        """
        return self._satellite.shell

    @property
    def ground_stations(self) -> tuple[GroundStation, ...]:
        """Return the static ground-station tuple from infrastructure."""
        return self._ground.ground_stations

    def snapshot_at(self, epoch: int, time_s: float) -> NetworkSnapshot:
        """Build a complete, frozen network snapshot.

        Args:
            epoch: Epoch index.
            time_s: TE time in seconds.
        """
        timeslot = self._satellite.time_to_timeslot(time_s)
        sat_state = self._satellite.state_at(timeslot)

        return NetworkSnapshot(
            epoch=epoch,
            time_s=time_s,
            satellite=sat_state,
            infra=self._infra_view,
        )
