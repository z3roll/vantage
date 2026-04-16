"""TEController factory and feedback protocol.

Controllers produce a :class:`RoutingPlane` per epoch.
Different strategies differ in how they compute cell→PoP assignments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vantage.world.ground import GroundKnowledge


@runtime_checkable
class SupportsGroundFeedback(Protocol):
    """Controller that uses ground delay feedback from GroundKnowledge."""

    @property
    def ground_knowledge(self) -> GroundKnowledge: ...


def create_controller(name: str, **kwargs: object):
    """Factory: create a controller by name.

    Args:
        name: Controller name ("nearest_pop", "ground_only", "static_pop",
              "greedy", "service_aware"). "latency_only" is an alias
              for "greedy".

    Returns:
        Controller instance with ``compute_routing_plane`` method.

    Raises:
        ValueError: If the controller name is unknown.
    """
    from vantage.control.policy.greedy import ProgressiveController
    from vantage.control.policy.ground_only import GroundOnlyController
    from vantage.control.policy.nearest_pop import NearestPoPController
    from vantage.control.policy.static_pop import StaticPoPController

    controllers: dict[str, type] = {
        "nearest_pop": NearestPoPController,
        "ground_only": GroundOnlyController,
        "static_pop": StaticPoPController,
        "progressive": ProgressiveController,
    }

    cls = controllers.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown controller: {name!r}. "
            f"Available: {sorted(controllers.keys())}"
        )
    return cls(**kwargs)  # type: ignore[no-any-return]
