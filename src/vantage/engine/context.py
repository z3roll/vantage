"""RunContext: shared runtime context for the TE system.

Bundles the globally-shared objects that flow through the epoch
loop:

- **world** — physical truth (satellite positions, ground infrastructure)
- **endpoints** — source/destination registry
- **ground_knowledge** — learned per-(pop, dest) RTT stats
  (mean/deviation/staleness) consumed by the controller during
  planning, updated by feedback after each epoch
- **ground_truth** — epoch-varying truth RTT sampler consumed by the
  data plane during ``forward.measure`` to produce the realized
  ``ground_rtt`` that feedback then absorbs into
  ``ground_knowledge``

Having ``ground_truth`` sit next to ``ground_knowledge`` on the
context — rather than hidden inside one or the other — keeps the
"plan from knowledge, measure from truth, learn truth into
knowledge" loop explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from vantage.domain import Endpoint
from vantage.world.ground import GroundKnowledge
from vantage.world.world import WorldModel

if TYPE_CHECKING:
    from vantage.world.ground import GroundTruth


@dataclass(frozen=True, slots=True)
class RunContext:
    """Shared runtime context for the TE engine.

    References are frozen (can't reassign), but the internal state of
    ``ground_knowledge`` (and optionally ``ground_truth``'s single-
    epoch memo) is mutable. ``ground_truth`` is optional because
    unit-test fixtures often don't need a real truth sampler — when
    ``None``, forward falls back to the planner's decided ground RTT
    in :meth:`forward.RoutingPlaneForward.measure`.

    simulation_start_utc: Absolute UTC time for epoch 0. Used when
    converting epochs into PoP-local time for time-varying models.
    """

    world: WorldModel
    endpoints: dict[str, Endpoint]
    ground_knowledge: GroundKnowledge
    simulation_start_utc: datetime | None = None
    ground_truth: "GroundTruth | None" = None
