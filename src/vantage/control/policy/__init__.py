"""Policy subsystem: TE controller strategies for PoP selection.

Each policy provides ``compute_routing_plane()`` which produces a
:class:`RoutingPlane` consumed by the data plane (forward.py).
"""

from vantage.control.policy.dualprice import DualPriceController
from vantage.control.policy.greedy import ProgressiveController
from vantage.control.policy.nearest_pop import NearestPoPController

__all__ = [
    "DualPriceController",
    "NearestPoPController",
    "ProgressiveController",
]
