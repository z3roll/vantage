"""Satellite segment: facade over constellation → topology → routing → visibility.

Usage:
    segment = SatelliteSegment(constellation, topology, shell_id, ground_stations, visibility)
    state = segment.state_at(timeslot=0)  # → SatelliteState (frozen, includes gateway attachments)

When use_tvg=True (default), uses TimeVaryingISLGraph for fast incremental
routing: fixed topology built once, only edge weights recomputed per timeslot.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType

from vantage.model.ground.infrastructure import GroundStation
from vantage.model.satellite.state import (
    AccessLink,
    GatewayAttachments,
    SatelliteState,
    ShellConfig,
)
from vantage.model.satellite.constellation import ConstellationModel
from vantage.model.satellite.routing import RoutingComputer, compute_all_pairs
from vantage.model.satellite.topology import PlusGridTopology, TopologyBuilder
from vantage.model.satellite.tvg import TimeVaryingISLGraph
from vantage.model.satellite.visibility import AccessModel


class SatelliteSegment:
    """Encapsulates the satellite network: constellation + ISL topology + routing + visibility.

    External callers see only ``state_at(timeslot)`` — the internal pipeline
    (positions → graph → routing → gateway attachments) is hidden.
    """

    def __init__(
        self,
        constellation: ConstellationModel,
        topology_builder: TopologyBuilder,
        shell_id: int,
        ground_stations: tuple[GroundStation, ...] = (),
        visibility: AccessModel | None = None,
        gateway_top_k: int = 8,
        routing: RoutingComputer | None = None,
        use_tvg: bool = True,
        state_cache_slots: int = 2,
    ) -> None:
        self._constellation = constellation
        self._topology = topology_builder
        self._shell_id = shell_id
        self._ground_stations = ground_stations
        self._visibility = visibility
        self._gateway_top_k = gateway_top_k
        self._routing: RoutingComputer = routing or compute_all_pairs
        if state_cache_slots < 1:
            raise ValueError(
                f"state_cache_slots must be >= 1, got {state_cache_slots}"
            )
        self._state_cache_slots = state_cache_slots

        # Validate and cache shell
        self._shell = self._get_shell()
        self._num_sats = self._shell.total_sats

        # Time-Varying Graph: build +Grid topology once, reuse across timeslots.
        # Only valid for PlusGridTopology — TVG hardcodes the +Grid structure.
        self._tvg: TimeVaryingISLGraph | None = None
        if use_tvg:
            if not isinstance(topology_builder, PlusGridTopology):
                raise TypeError(
                    f"use_tvg=True requires PlusGridTopology, got {type(topology_builder).__name__}"
                )
            self._tvg = TimeVaryingISLGraph(self._shell)

        # Timeslot → SatelliteState cache. Keep only a tiny LRU window:
        # the simulator advances at 1 s but constellation timeslots are
        # ~15 s apart, so retaining the current/adjacent slots preserves
        # reuse while preventing unbounded growth across a long run.
        self._state_cache: OrderedDict[int, SatelliteState] = OrderedDict()

    @property
    def shell(self) -> ShellConfig:
        """Return the :class:`ShellConfig` this segment operates on.

        Cached at construction via :meth:`_get_shell`; exposed as a
        public read-only property so callers (capacity views, multi-
        shell aware policies, etc.) don't have to reach into private
        state.
        """
        return self._shell

    @property
    def ground_stations(self) -> tuple[GroundStation, ...]:
        """Static ground-station tuple this segment was configured with."""
        return self._ground_stations

    def state_at(self, timeslot: int) -> SatelliteState:
        """Compute full satellite segment state at a given timeslot.

        With TVG enabled: positions → vectorized weight update → scipy Dijkstra.
        Without TVG: positions → topology build → routing backend.
        Results are cached in a small LRU window so repeated queries for
        the active timeslot stay O(1) without retaining every historical
        routing matrix for the whole run.
        """
        cached = self._state_cache.get(timeslot)
        if cached is not None:
            self._state_cache.move_to_end(timeslot)
            return cached

        positions = self._constellation.positions_array_at(timeslot, self._shell_id)

        if self._tvg is not None:
            # Fast path: fixed topology + scipy all-pairs
            graph, routing = self._tvg.compute_state(positions, timeslot)
            delay_matrix = routing.delay_matrix
            predecessor_matrix = routing.predecessor_matrix
        else:
            # Legacy path: full rebuild + all-pairs routing
            graph = self._topology.build(self._shell, positions, timeslot)
            routing = self._routing(graph)
            delay_matrix = routing.delay_matrix
            predecessor_matrix = routing.predecessor_matrix

        gw = self._compute_gateway_attachments(positions)

        state = SatelliteState(
            positions=positions,
            graph=graph,
            delay_matrix=delay_matrix,
            predecessor_matrix=predecessor_matrix,
            gateway_attachments=gw,
        )
        self._state_cache[timeslot] = state
        self._state_cache.move_to_end(timeslot)
        while len(self._state_cache) > self._state_cache_slots:
            self._state_cache.popitem(last=False)
        return state

    def _compute_gateway_attachments(self, positions):
        """Compute top-k visible satellites for each enabled ground station."""
        if not self._ground_stations or self._visibility is None:
            return GatewayAttachments(attachments={})

        enabled_gs = [gs for gs in self._ground_stations if gs.enabled]
        vis = self._visibility
        top_k = self._gateway_top_k

        def _compute_one(gs: GroundStation) -> tuple[str, tuple[AccessLink, ...]] | None:
            visible = vis.compute_access(
                gs.lat_deg, gs.lon_deg, 0.0, positions, gs.min_elevation_deg,
            )
            links = visible[:top_k]
            return (gs.gs_id, links) if links else None

        attachments: dict[str, tuple[AccessLink, ...]] = {}
        with ThreadPoolExecutor() as pool:
            for result in pool.map(_compute_one, enabled_gs):
                if result is not None:
                    attachments[result[0]] = result[1]

        return GatewayAttachments(attachments=MappingProxyType(attachments))

    @property
    def shell_id(self) -> int:
        return self._shell_id

    @property
    def num_sats(self) -> int:
        return self._num_sats

    @property
    def num_timeslots(self) -> int:
        return self._constellation.num_timeslots_for_shell(self._shell_id)

    @property
    def dt_s(self) -> float:
        """Time step between constellation timeslots (seconds)."""
        return self._shell.orbit_cycle_s / self.num_timeslots

    def time_to_timeslot(self, time_s: float) -> int:
        """Convert simulation time to timeslot index (wraps at orbit period)."""
        ts = int(time_s / self.dt_s) % self.num_timeslots
        return ts

    def _get_shell(self) -> ShellConfig:
        for shell in self._constellation.config.shells:
            if shell.shell_id == self._shell_id:
                return shell
        raise ValueError(f"Shell {self._shell_id} not found in constellation config")
