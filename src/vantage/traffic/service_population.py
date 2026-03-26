"""Service-class population model: terminals -> service classes."""

from __future__ import annotations

import json
from pathlib import Path

from vantage.domain import Endpoint
from vantage.domain.service import SERVICE_CLASSES


class ServiceClassPopulation:
    """Population with service-class destinations instead of geographic endpoints.

    Sources are user terminals (same as EndpointPopulation).
    Destinations are abstract service classes (no lat/lon).
    """

    def __init__(
        self,
        sources: tuple[Endpoint, ...],
        service_classes: tuple[str, ...] = SERVICE_CLASSES,
    ) -> None:
        self._sources = sources
        self._service_classes = service_classes

    @property
    def sources(self) -> tuple[Endpoint, ...]:
        return self._sources

    @property
    def service_classes(self) -> tuple[str, ...]:
        return self._service_classes

    @staticmethod
    def from_terminal_registry(
        terminals_path: str | Path,
        service_classes: tuple[str, ...] | None = None,
    ) -> ServiceClassPopulation:
        """Load source terminals from terminal registry.

        Args:
            terminals_path: Path to terminals.json.
            service_classes: Override default service classes.
        """
        with Path(terminals_path).open() as f:
            terminals = json.load(f)

        sources = tuple(
            Endpoint(
                name=f"terminal_{t['terminal_id']}",
                lat_deg=t["lat_deg"],
                lon_deg=t["lon_deg"],
            )
            for t in terminals
        )

        if service_classes is None:
            service_classes = SERVICE_CLASSES

        return ServiceClassPopulation(sources, service_classes)
