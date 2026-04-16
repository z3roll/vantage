"""Endpoint population: source terminals, destinations, and city grouping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from vantage.domain import Endpoint


@dataclass(frozen=True, slots=True)
class CityGroup:
    """A city with its Starlink user count and assigned terminal names."""

    city: str
    country: str
    lat: float
    lon: float
    user_count: int
    terminal_names: tuple[str, ...]


class EndpointPopulation:
    """Defines source endpoints (user terminals) and destination endpoints."""

    def __init__(
        self,
        sources: tuple[Endpoint, ...],
        destinations: tuple[Endpoint, ...],
        city_groups: tuple[CityGroup, ...] = (),
    ) -> None:
        self._sources = sources
        self._destinations = destinations
        self._city_groups = city_groups

    @property
    def sources(self) -> tuple[Endpoint, ...]:
        return self._sources

    @property
    def destinations(self) -> tuple[Endpoint, ...]:
        return self._destinations

    @property
    def city_groups(self) -> tuple[CityGroup, ...]:
        return self._city_groups

    @staticmethod
    def from_terminal_registry(
        terminals_path: str | Path,
        destinations: tuple[Endpoint, ...] | None = None,
    ) -> EndpointPopulation:
        """Load source terminals from the terminal registry."""
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
            raise ValueError("destinations must be provided (no hardcoded defaults)")

        return EndpointPopulation(sources, destinations)

    @staticmethod
    def from_starlink_users(
        users_path: str | Path,
        cities_path: str | Path,
        destinations: tuple[Endpoint, ...] | None = None,
        *,
        user_scale: float = 1.0,
    ) -> EndpointPopulation:
        """Generate one endpoint per city, with user counts from Starlink data.

        Each city with Starlink users becomes one routing endpoint.
        The city's user count is stored in :class:`CityGroup` for
        the per-city Poisson arrival model.

        Args:
            user_scale: Multiplier on user counts. 1.0 = real Starlink
                data (~6.4M users). 0.01 = 1% sample (~64K users).
        """
        with Path(users_path).open() as f:
            country_users: dict[str, int] = json.load(f)
        with Path(cities_path).open() as f:
            all_cities: list[dict] = json.load(f)

        codes_path = Path(users_path).parent / "country_codes.json"
        with codes_path.open() as f:
            name_to_iso: dict[str, str] = json.load(f)

        cities_by_code: dict[str, list[dict]] = {}
        for c in all_cities:
            cities_by_code.setdefault(c["country"], []).append(c)

        sources: list[Endpoint] = []
        groups: list[CityGroup] = []

        for country_name, n_users in country_users.items():
            code = name_to_iso.get(country_name)
            if code is None:
                continue
            cities = cities_by_code.get(code, [])
            if not cities:
                continue

            city_weights = [c.get("weight", 1.0) for c in cities]
            total_w = sum(city_weights)
            if total_w <= 0:
                total_w = len(cities)
                city_weights = [1.0] * len(cities)

            for ci, city in enumerate(cities):
                city_user_count = max(1, round(n_users * user_scale * city_weights[ci] / total_w))
                terminal_name = f"city_{code}_{city['city'].replace(' ', '_')}"

                sources.append(Endpoint(
                    name=terminal_name,
                    lat_deg=city["lat"],
                    lon_deg=city["lon"],
                ))
                groups.append(CityGroup(
                    city=city["city"],
                    country=code,
                    lat=city["lat"],
                    lon=city["lon"],
                    user_count=city_user_count,
                    terminal_names=(terminal_name,),
                ))

        if destinations is None:
            raise ValueError("destinations must be provided (no hardcoded defaults)")

        return EndpointPopulation(
            tuple(sources), destinations, tuple(groups),
        )
