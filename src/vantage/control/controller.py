"""TEController Protocol and factory.

Controllers precompute cost tables (sat_cost + ground_cost) per epoch.
Terminal-side PoP selection is done in forward.py using these tables.
Different strategies differ only in how they fill the cost tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from vantage.domain import CostTables, NetworkSnapshot

if TYPE_CHECKING:
    from vantage.world.ground import GroundKnowledge


class TEController(Protocol):
    """Traffic engineering controller interface.

    Controllers compute cost tables from the current network state.
    These tables are broadcast to terminals, which locally select
    the best PoP via argmin(sat_cost + ground_cost).
    """

    def compute_tables(
        self,
        snapshot: NetworkSnapshot,
    ) -> CostTables:
        """Precompute satellite and ground cost tables.

        Args:
            snapshot: Current physical network state.

        Returns:
            CostTables with sat_cost and ground_cost for terminal-side selection.
        """
        ...


@runtime_checkable
class SupportsGroundFeedback(Protocol):
    """Controller that uses ground delay feedback from GroundKnowledge."""

    @property
    def ground_knowledge(self) -> GroundKnowledge: ...


def create_controller(name: str, **kwargs: object) -> TEController:
    """Factory: create a controller by name.

    Args:
        name: Controller name ("nearest_pop", "ground_only", "static_pop",
              "greedy"). "latency_only" is an alias for "greedy".

    Returns:
        TEController instance.

    Raises:
        ValueError: If the controller name is unknown.
    """
    from vantage.control.policy.greedy import VantageGreedyController
    from vantage.control.policy.ground_only import GroundOnlyController
    from vantage.control.policy.nearest_pop import NearestPoPController
    from vantage.control.policy.service_aware import ServiceAwareController
    from vantage.control.policy.static_pop import StaticPoPController

    controllers: dict[str, type] = {
        "nearest_pop": NearestPoPController,
        "ground_only": GroundOnlyController,
        "static_pop": StaticPoPController,
        "latency_only": VantageGreedyController,
        "greedy": VantageGreedyController,
        "service_aware": ServiceAwareController,
    }

    cls = controllers.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown controller: {name!r}. "
            f"Available: {sorted(controllers.keys())}"
        )
    return cls(**kwargs)  # type: ignore[no-any-return]
