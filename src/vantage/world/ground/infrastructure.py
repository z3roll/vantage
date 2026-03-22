"""Ground infrastructure: loads pre-processed static data.

No parsing logic — reads clean JSON produced by config/preprocess.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from vantage.domain import GroundStation, GSPoPEdge, PoP

_T = TypeVar("_T", GroundStation, PoP, GSPoPEdge)

EXPECTED_SCHEMA_VERSION = 1


class GroundInfrastructure:
    """Loads and holds pre-processed static ground infrastructure data.

    Validates manifest schema_version on load.
    """

    def __init__(self, data_dir: str | Path) -> None:
        data_dir = Path(data_dir)
        self._validate_manifest(data_dir)

        self._pops = self._load(data_dir / "pops.json", PoP)
        self._ground_stations = self._load(
            data_dir / "ground_stations.json", GroundStation
        )
        self._gs_pop_edges = self._load(
            data_dir / "gs_pop_edges.json", GSPoPEdge
        )

        self._pop_by_code: dict[str, PoP] = {p.code: p for p in self._pops}
        self._gs_by_id: dict[str, GroundStation] = {
            g.gs_id: g for g in self._ground_stations
        }

        # GS → connected PoPs, PoP → connected GSs
        self._gs_to_pops: dict[str, list[GSPoPEdge]] = {}
        self._pop_to_gs: dict[str, list[GSPoPEdge]] = {}
        for edge in self._gs_pop_edges:
            self._gs_to_pops.setdefault(edge.gs_id, []).append(edge)
            self._pop_to_gs.setdefault(edge.pop_code, []).append(edge)

    @staticmethod
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

    @staticmethod
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

    @property
    def pops(self) -> tuple[PoP, ...]:
        return self._pops

    @property
    def ground_stations(self) -> tuple[GroundStation, ...]:
        return self._ground_stations

    @property
    def gs_pop_edges(self) -> tuple[GSPoPEdge, ...]:
        return self._gs_pop_edges

    def pop_by_code(self, code: str) -> PoP | None:
        return self._pop_by_code.get(code)

    def gs_by_id(self, gs_id: str) -> GroundStation | None:
        return self._gs_by_id.get(gs_id)

    def gs_serving_pop(self, pop_code: str) -> tuple[GSPoPEdge, ...]:
        """Return all GS↔PoP edges for a given PoP."""
        return tuple(self._pop_to_gs.get(pop_code, []))

    def pops_reachable_from_gs(self, gs_id: str) -> tuple[GSPoPEdge, ...]:
        """Return all GS↔PoP edges for a given ground station."""
        return tuple(self._gs_to_pops.get(gs_id, []))
