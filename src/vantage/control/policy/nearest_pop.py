"""NearestPoP controller: hot-potato baseline.

Fills sat_cost with real values, ground_cost with zeros.
Terminal picks PoP with lowest satellite cost = nearest PoP.
"""

from __future__ import annotations

from types import MappingProxyType

from vantage.control.policy.common.sat_cost import precompute_sat_cost
from vantage.domain import CostTables, NetworkSnapshot


class NearestPoPController:
    """Baseline: route to PoP with lowest satellite segment cost (nearest)."""

    def compute_tables(self, snapshot: NetworkSnapshot) -> CostTables:
        sat_cost = precompute_sat_cost(snapshot)
        # Ground cost = 0 for all → terminal picks purely by satellite cost
        ground_cost: dict[tuple[str, str], float] = {}
        return CostTables(
            epoch=snapshot.epoch,
            sat_cost=MappingProxyType(sat_cost),
            ground_cost=MappingProxyType(ground_cost),
        )
