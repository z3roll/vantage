"""Endpoint population: source terminals and destinations."""

from __future__ import annotations

import json
from pathlib import Path

from vantage.domain import Endpoint


# Well-known CDN/server locations
DEFAULT_DESTINATIONS: tuple[Endpoint, ...] = (
    Endpoint("google", 37.4220, -122.0841),       # Mountain View, CA
    Endpoint("facebook", 37.4845, -122.1477),      # Menlo Park, CA
    Endpoint("wikipedia", 38.8951, -77.0364),       # Ashburn, VA
    Endpoint("cloudflare", 37.7749, -122.4194),     # San Francisco, CA
    Endpoint("amazon", 47.6062, -122.3321),         # Seattle, WA
)


class EndpointPopulation:
    """Defines source endpoints (user terminals) and destination endpoints.

    Sources are loaded from the terminal registry. Destinations are
    well-known server locations (Google, Facebook, etc.).
    """

    def __init__(
        self,
        sources: tuple[Endpoint, ...],
        destinations: tuple[Endpoint, ...],
    ) -> None:
        self._sources = sources
        self._destinations = destinations

    @property
    def sources(self) -> tuple[Endpoint, ...]:
        return self._sources

    @property
    def destinations(self) -> tuple[Endpoint, ...]:
        return self._destinations

    @staticmethod
    def from_terminal_registry(
        terminals_path: str | Path,
        destinations: tuple[Endpoint, ...] | None = None,
    ) -> EndpointPopulation:
        """Load source terminals from the terminal registry.

        Args:
            terminals_path: Path to terminals.json (processed data).
            destinations: Destination endpoints. Defaults to
                well-known server locations.
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

        if destinations is None:
            destinations = DEFAULT_DESTINATIONS

        return EndpointPopulation(sources, destinations)
