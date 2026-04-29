"""Ground infrastructure model and JSON loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar


@dataclass(frozen=True, slots=True)
class GroundStation:
    """A Starlink ground station gateway."""

    gs_id: str
    lat_deg: float
    lon_deg: float
    country: str
    town: str
    num_antennas: int
    min_elevation_deg: float
    enabled: bool
    uplink_ghz: float
    downlink_ghz: float
    max_capacity: float
    temporary: bool
    ka_antennas: int = 0
    e_antennas: int = 0


@dataclass(frozen=True, slots=True)
class PoP:
    """A Starlink Point of Presence."""

    site_id: str
    code: str
    name: str
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True, slots=True)
class GSPoPEdge:
    """Static fiber connection between a ground station and a PoP."""

    gs_id: str
    pop_code: str
    distance_km: float
    backhaul_delay: float
    capacity_gbps: float


_T = TypeVar("_T", GroundStation, PoP, GSPoPEdge)

EXPECTED_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class GroundInfrastructure:
    """Static ground infrastructure plus lookup indexes."""

    pops: tuple[PoP, ...]
    ground_stations: tuple[GroundStation, ...]
    gs_pop_edges: tuple[GSPoPEdge, ...]
    _pop_by_code: dict[str, PoP] = field(init=False, repr=False, compare=False)
    _gs_by_id: dict[str, GroundStation] = field(init=False, repr=False, compare=False)
    _backhaul_map: dict[tuple[str, str], float] = field(init=False, repr=False, compare=False)
    _gs_to_pops: dict[str, list[GSPoPEdge]] = field(init=False, repr=False, compare=False)
    _pop_to_gs: dict[str, list[GSPoPEdge]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        pop_by_code = {p.code: p for p in self.pops}
        gs_by_id = {g.gs_id: g for g in self.ground_stations}
        backhaul = {(e.gs_id, e.pop_code): e.backhaul_delay for e in self.gs_pop_edges}
        gs_to_pops: dict[str, list[GSPoPEdge]] = {}
        pop_to_gs: dict[str, list[GSPoPEdge]] = {}
        for edge in self.gs_pop_edges:
            gs_to_pops.setdefault(edge.gs_id, []).append(edge)
            pop_to_gs.setdefault(edge.pop_code, []).append(edge)
        object.__setattr__(self, "_pop_by_code", pop_by_code)
        object.__setattr__(self, "_gs_by_id", gs_by_id)
        object.__setattr__(self, "_backhaul_map", backhaul)
        object.__setattr__(self, "_gs_to_pops", gs_to_pops)
        object.__setattr__(self, "_pop_to_gs", pop_to_gs)

    @classmethod
    def from_config(cls, data_dir: str | Path) -> GroundInfrastructure:
        data_dir = Path(data_dir)
        _validate_manifest(data_dir)
        return cls(
            pops=_load(data_dir / "pops.json", PoP),
            ground_stations=_load(data_dir / "ground_stations.json", GroundStation),
            gs_pop_edges=_load(data_dir / "gs_pop_edges.json", GSPoPEdge),
        )

    def with_gs_pop_edges(self, edges: tuple[GSPoPEdge, ...]) -> GroundInfrastructure:
        """Return a copy with a different GS↔PoP edge set."""
        return GroundInfrastructure(
            pops=self.pops,
            ground_stations=self.ground_stations,
            gs_pop_edges=tuple(edges),
        )

    def pop_by_code(self, code: str) -> PoP | None:
        return self._pop_by_code.get(code)

    def gs_by_id(self, gs_id: str) -> GroundStation | None:
        return self._gs_by_id.get(gs_id)

    def get_backhaul_delay(self, gs_id: str, pop_code: str) -> float:
        return self._backhaul_map.get((gs_id, pop_code), 0.0)

    def pop_gs_edges(self, pop_code: str) -> tuple[tuple[str, float], ...]:
        return tuple((e.gs_id, e.backhaul_delay) for e in self._pop_to_gs.get(pop_code, ()))

    def gs_serving_pop(self, pop_code: str) -> tuple[GSPoPEdge, ...]:
        return tuple(self._pop_to_gs.get(pop_code, ()))

    def pops_reachable_from_gs(self, gs_id: str) -> tuple[GSPoPEdge, ...]:
        return tuple(self._gs_to_pops.get(gs_id, ()))


def _validate_manifest(data_dir: Path) -> None:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {data_dir}. "
            "Run: uv run python -m vantage.config.preprocess"
        )
    with manifest_path.open() as f:
        manifest = json.load(f)
    version = manifest.get("schema_version", 0)
    if version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: expected {EXPECTED_SCHEMA_VERSION}, "
            f"got {version}. Re-run: uv run python -m vantage.config.preprocess"
        )


def _load(path: Path, cls: type[_T]) -> tuple[_T, ...]:
    try:
        with path.open() as f:
            records = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Processed data file not found: {path}. "
            "Run: uv run python -m vantage.config.preprocess"
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt JSON in {path}: {exc}") from exc
    try:
        return tuple(cls(**entry) for entry in records)
    except TypeError as exc:
        raise ValueError(f"Schema mismatch in {path}: {exc}") from exc
