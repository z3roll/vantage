"""RunContext: shared runtime context for the TE system.

Bundles the three globally-shared objects that flow through the
epoch loop:

- **world** — physical truth (satellite positions, ground infrastructure)
- **endpoints** — source/destination registry
- **ground_knowledge** — unified ground delay service (cache + estimator)

All consumers (controller, forward, engine feedback) depend on
ground_knowledge as the single source of truth for ground delays.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from vantage.domain import Endpoint
from vantage.world.ground import GroundKnowledge
from vantage.world.world import WorldModel


@dataclass(frozen=True, slots=True)
class RunContext:
    """Shared runtime context for the TE engine.

    References are frozen (can't reassign), but internal state of
    ground_knowledge is mutable (populated by engine feedback loop).
    Satellite delay calibration is owned by WorldModel.

    simulation_start_utc: Absolute UTC time for epoch 0. Used when
    converting epochs into PoP-local time for time-varying models.
    """

    world: WorldModel
    endpoints: dict[str, Endpoint]
    ground_knowledge: GroundKnowledge
    simulation_start_utc: datetime | None = None
