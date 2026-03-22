"""TEController Protocol and factory.

Controllers produce fully resolved PathAllocations (pop, gs, user_sat,
egress_sat). The forwarding engine computes actual delays — no search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from vantage.domain import NetworkSnapshot, RoutingIntent, TrafficDemand

if TYPE_CHECKING:
    from vantage.world.ground import GroundKnowledge


class TEController(Protocol):
    """Traffic engineering controller interface.

    All controllers implement this. One config line swaps the algorithm.
    Controllers output fully resolved PathAllocations.
    """

    def optimize(
        self,
        snapshot: NetworkSnapshot,
        demand: TrafficDemand,
    ) -> RoutingIntent:
        """Produce path allocations for each flow.

        Args:
            snapshot: Current physical network state (truth).
            demand: Traffic demand for this epoch.

        Returns:
            RoutingIntent with fully resolved PathAllocation per flow.
        """
        ...


@runtime_checkable
class SupportsGroundFeedback(Protocol):
    """Optional Protocol for controllers that use ground delay feedback.

    Controllers implementing this read from a GroundKnowledge service
    during optimize(). The engine populates context.ground_knowledge
    with observed ground delays when this Protocol is detected.
    The controller must be constructed with the *same* GroundKnowledge
    instance as context.ground_knowledge.
    """

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        """The ground knowledge service this controller reads from."""
        ...


# Backward-compatible alias
SupportsGroundDelayFeedback = SupportsGroundFeedback


def create_controller(name: str, **kwargs: object) -> TEController:
    """Factory: create a controller by name.

    Each controller outputs fully resolved PathAllocations. The only
    difference between controllers is the PoP selection strategy.

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
    from vantage.control.policy.static_pop import StaticPoPController

    controllers: dict[str, type] = {
        "nearest_pop": NearestPoPController,
        "ground_only": GroundOnlyController,
        "static_pop": StaticPoPController,
        "latency_only": VantageGreedyController,
        "greedy": VantageGreedyController,
    }

    cls = controllers.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown controller: {name!r}. "
            f"Available: {sorted(controllers.keys())}"
        )
    return cls(**kwargs)  # type: ignore[no-any-return]
