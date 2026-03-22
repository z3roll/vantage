"""Path resolver: converts high-level PoP decisions into fully resolved paths.

Sits between controller (PoP selection) and forward (delay computation).
Currently a thin passthrough — controllers already produce full PathAllocations.
As the system evolves toward controller-outputs-PoP-only, this layer will
take over GS + satellite resolution via the candidate framework.
"""

from __future__ import annotations

from typing import Protocol

from vantage.domain.result import RoutingIntent
from vantage.domain.snapshot import NetworkSnapshot


class PathResolver(Protocol):
    """Protocol for resolving routing intent into executable paths.

    Current system: identity resolver (controllers already produce
    fully resolved paths).
    Future: resolve PoP decisions → (GS, sat) paths using candidate
    enumeration.
    """

    def resolve(
        self,
        intent: RoutingIntent,
        snapshot: NetworkSnapshot,
    ) -> RoutingIntent:
        """Resolve or refine a routing intent.

        May add GS/satellite details, apply capacity constraints,
        or split flows across multiple paths.
        """
        ...


class IdentityResolver:
    """Passthrough resolver: returns intent unchanged.

    Used when controllers already produce fully resolved PathAllocations.
    """

    def resolve(
        self,
        intent: RoutingIntent,
        snapshot: NetworkSnapshot,
    ) -> RoutingIntent:
        return intent
