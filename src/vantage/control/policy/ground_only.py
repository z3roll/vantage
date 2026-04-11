"""GroundOnly controller: ground-delay oracle baseline.

Fills ``sat_cost`` with real values (so the forward plane still has
valid ``(ingress, pop)`` entries) and ``ground_cost`` from a
:class:`GroundDelay` measurement table. Pairs without measurements
are silently omitted from the ``ground_cost`` map — the terminal
will then be unable to select them, which is the strict "no
geographic fallback" semantics the project now enforces.
"""

from __future__ import annotations

import logging
from types import MappingProxyType

from vantage.control.policy.common.sat_cost import precompute_sat_cost
from vantage.domain import CostTables, Endpoint, NetworkSnapshot
from vantage.world.ground import GroundDelay, MeasuredGroundDelay

_log = logging.getLogger(__name__)


class GroundOnlyController:
    """Baseline: route to the PoP with lowest *measured* ground delay.

    Only ``(pop, dest)`` pairs present in the supplied
    :class:`GroundDelay` table enter ``ground_cost``. No geographic
    estimation is performed; unknown pairs are omitted.

    When constructed without a ``ground_delay`` (e.g. through the
    ``create_controller`` factory with no extra kwargs), the
    controller defaults to :meth:`MeasuredGroundDelay.empty`, which
    raises on every lookup. The ``ground_cost`` table is then empty
    and the forward plane falls back to ``sat_cost``-only selection
    (equivalent to nearest-PoP). That's the intended "no data
    provided" semantics — the policy exists but has nothing to say.
    """

    def __init__(
        self,
        ground_delay: GroundDelay | None = None,
        endpoints: dict[str, Endpoint] | None = None,
    ) -> None:
        self._endpoints = endpoints or {}
        self._ground_delay: GroundDelay = ground_delay or MeasuredGroundDelay.empty()
        # An empty table degrades this policy to nearest-PoP silently.
        # Log a warning so operators see it in production runs.
        if (
            isinstance(self._ground_delay, MeasuredGroundDelay)
            and len(self._ground_delay) == 0
        ):
            _log.warning(
                "GroundOnlyController: ground_delay table is empty; "
                "ground_cost will be empty and the controller will "
                "degrade to sat_cost-only (nearest-PoP) selection."
            )

    def compute_tables(self, snapshot: NetworkSnapshot) -> CostTables:
        # Sat cost populated so forward.py has valid (ingress, pop) entries.
        sat_cost = precompute_sat_cost(snapshot)

        # Ground cost: only measured pairs. Missing (pop, dest) pairs
        # are skipped so the terminal's argmin never sees them.
        ground_cost: dict[tuple[str, str], float] = {}
        destinations = [
            ep for ep in self._endpoints.values()
            if not ep.name.startswith("terminal_")
        ]
        for pop in snapshot.infra.pops:
            for dst in destinations:
                try:
                    one_way = self._ground_delay.estimate(pop.code, dst.name)
                except KeyError:
                    continue
                ground_cost[(pop.code, dst.name)] = one_way * 2  # RTT

        return CostTables(
            epoch=snapshot.epoch,
            sat_cost=MappingProxyType(sat_cost),
            ground_cost=MappingProxyType(ground_cost),
        )
