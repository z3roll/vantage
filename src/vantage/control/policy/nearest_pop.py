"""NearestPoP controller: hot-potato baseline.

This controller has two execution modes, both kept in lock-step so
the baseline can be compared against new policies during the
transition:

    * :meth:`compute_tables` — legacy output (``CostTables``).
      Fills ``sat_cost`` with real values and ``ground_cost`` with an
      empty map so the terminal-side ``argmin`` degenerates to
      "nearest PoP by satellite cost". This is what
      :mod:`vantage.forward` currently consumes.

    * :meth:`compute_routing_plane` — new output (``RoutingPlane``).
      Expresses the same nearest-PoP decision as a two-table forwarding
      plane pushed to every satellite every 15 seconds:
      a static geographic ``cell → pop`` map plus a per-sat
      ``pop → next_hop`` FIB derived from the existing Dijkstra output.
      This matches the PPT Slide 13 control-plane narrative and is the
      input format the next forward.py revision will consume.

Both methods return views of the *same* underlying routing decisions,
so any discrepancy between the two is a bug.
"""

from __future__ import annotations

from types import MappingProxyType

from vantage.control.policy.common.fib_builder import (
    build_routing_plane_nearest_pop,
)
from vantage.control.policy.common.sat_cost import precompute_sat_cost
from vantage.domain import CellGrid, CostTables, NetworkSnapshot, RoutingPlane


class NearestPoPController:
    """Baseline: route to the PoP with the lowest satellite-segment cost."""

    def compute_tables(self, snapshot: NetworkSnapshot) -> CostTables:
        """Legacy CostTables output — consumed by the current forward.py."""
        sat_cost = precompute_sat_cost(snapshot)
        # Ground cost = 0 for all → terminal picks purely by satellite cost.
        ground_cost: dict[tuple[str, str], float] = {}
        return CostTables(
            epoch=snapshot.epoch,
            sat_cost=MappingProxyType(sat_cost),
            ground_cost=MappingProxyType(ground_cost),
        )

    def compute_routing_plane(
        self,
        snapshot: NetworkSnapshot,
        cell_grid: CellGrid,
        *,
        version: int = 0,
    ) -> RoutingPlane:
        """New RoutingPlane output — used by the forthcoming forward.py path.

        The plan is computed by composing two policy-agnostic helpers:

            * ``build_cell_to_pop_nearest`` — geographic argmin, baseline.
            * ``build_satellite_fibs`` — reuses the Dijkstra result that
              already lives in ``snapshot.satellite.predecessor_matrix``,
              so no extra routing computation is spent here.

        The two helpers are free functions (not methods of this class)
        so future policies can reuse them without inheriting from
        ``NearestPoPController``.
        """
        return build_routing_plane_nearest_pop(
            snapshot=snapshot,
            cell_grid=cell_grid,
            version=version,
        )
