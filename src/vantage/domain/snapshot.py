"""Network snapshot domain types.

Composes satellite and ground segments into a complete frozen view.
All delay values in ms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from vantage.domain.ground import GroundStation, GSPoPEdge, PoP
from vantage.domain.satellite import SatelliteState


@dataclass(frozen=True, slots=True)
class InfrastructureView:
    """Frozen view of static ground infrastructure with pre-built indexes."""

    pops: tuple[PoP, ...]
    ground_stations: tuple[GroundStation, ...]
    gs_pop_edges: tuple[GSPoPEdge, ...]

    # Pre-built indexes (init=False, set in __post_init__)
    _pop_map: dict[str, PoP] = field(
        init=False, repr=False, compare=False
    )
    _gs_map: dict[str, GroundStation] = field(
        init=False, repr=False, compare=False
    )
    _backhaul_map: dict[tuple[str, str], float] = field(
        init=False, repr=False, compare=False
    )
    _pop_gs_map: dict[str, list[tuple[str, float]]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_pop_map", {p.code: p for p in self.pops}
        )
        object.__setattr__(
            self, "_gs_map", {g.gs_id: g for g in self.ground_stations}
        )
        object.__setattr__(
            self,
            "_backhaul_map",
            {(e.gs_id, e.pop_code): e.backhaul_delay for e in self.gs_pop_edges},
        )
        pop_gs: dict[str, list[tuple[str, float]]] = {}
        for e in self.gs_pop_edges:
            pop_gs.setdefault(e.pop_code, []).append(
                (e.gs_id, e.backhaul_delay)
            )
        object.__setattr__(self, "_pop_gs_map", pop_gs)

    def pop_by_code(self, code: str) -> PoP | None:
        """Lookup PoP by code."""
        return self._pop_map.get(code)

    def gs_by_id(self, gs_id: str) -> GroundStation | None:
        """Lookup ground station by ID."""
        return self._gs_map.get(gs_id)

    def get_backhaul_delay(self, gs_id: str, pop_code: str) -> float:
        """One-way backhaul delay (ms) for a GS↔PoP pair. Returns 0.0 if not found."""
        return self._backhaul_map.get((gs_id, pop_code), 0.0)

    def pop_gs_edges(self, pop_code: str) -> tuple[tuple[str, float], ...]:
        """Return ((gs_id, backhaul_delay_ms), ...) for a given PoP."""
        return tuple(self._pop_gs_map.get(pop_code, []))


@dataclass(frozen=True, slots=True)
class NetworkSnapshot:
    """Complete physical network state at time t. Fully frozen."""

    epoch: int
    time_s: float
    satellite: SatelliteState
    infra: InfrastructureView
