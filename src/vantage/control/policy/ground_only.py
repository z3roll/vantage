"""GroundOnly controller: ground-delay oracle baseline.

Fills sat_cost with zeros, ground_cost with L2 estimates.
Terminal picks PoP with lowest ground cost = nearest to destination.
"""

from __future__ import annotations

from types import MappingProxyType

from vantage.control.policy.common.sat_cost import precompute_sat_cost
from vantage.domain import CostTables, Endpoint, NetworkSnapshot
from vantage.world.ground import GroundDelay, HaversineDelay


class GroundOnlyController:
    """Baseline: route to PoP with lowest ground delay (ignoring satellite)."""

    def __init__(
        self,
        endpoints: dict[str, Endpoint] | None = None,
        ground_delay: GroundDelay | None = None,
    ) -> None:
        self._endpoints = endpoints or {}
        self._ground_delay: GroundDelay = ground_delay or HaversineDelay()

    def compute_tables(self, snapshot: NetworkSnapshot) -> CostTables:
        # Sat cost populated but not used for ranking — ground cost dominates.
        # Still needed so forward.py can find valid (ingress, pop) entries.
        sat_cost = precompute_sat_cost(snapshot)

        # Ground cost = L2 estimate for all (PoP, dest) pairs
        ground_cost: dict[tuple[str, str], float] = {}
        destinations = [
            ep for ep in self._endpoints.values()
            if not ep.name.startswith("terminal_")
        ]
        for pop in snapshot.infra.pops:
            for dst in destinations:
                ground_cost[(pop.code, dst.name)] = self._ground_delay.estimate(
                    pop.lat_deg, pop.lon_deg, dst.lat_deg, dst.lon_deg
                ) * 2  # RTT

        return CostTables(
            epoch=snapshot.epoch,
            sat_cost=MappingProxyType(sat_cost),
            ground_cost=MappingProxyType(ground_cost),
        )
