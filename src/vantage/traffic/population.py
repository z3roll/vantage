"""Endpoint population: source terminals, destinations, and city grouping."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import h3

from vantage.traffic.types import Endpoint

# Target users per sub-endpoint. Big cities (NY ~830k Starlink users at
# scale=5) split into ~ceil(user_count / TARGET) sub-endpoints, so a
# single (sub-endpoint, service) flow's per-epoch demand stays within
# the per-sat 20 Gbps Ka feeder cap and the cascade can route it
# without falling into the defensive overflow tail. Empirically chosen
# so NY's biggest service (google) lands at <5 Gbps per sub-endpoint.
_USERS_PER_SUB_ENDPOINT: int = 20_000

# Hard ceiling on sub-endpoints per city — guards against unbounded
# growth if an upstream config bumps user_scale by orders of magnitude.
_MAX_SUB_ENDPOINTS_PER_CITY: int = 64

# H3 resolution of the cell grid used for routing decisions; sub-endpoint
# centroids are picked from this resolution's grid disk so each
# sub-endpoint typically lands in its own routing cell.
_SUB_ENDPOINT_H3_RES: int = 5


def _split_factor(user_count: int) -> int:
    """How many sub-endpoints a city of this size should split into."""
    n = math.ceil(user_count / _USERS_PER_SUB_ENDPOINT)
    return max(1, min(n, _MAX_SUB_ENDPOINTS_PER_CITY))


def _spread_subcells(
    lat: float, lon: float, n_sub: int,
) -> list[tuple[float, float]]:
    """Pick ``n_sub`` H3 cell centroids around ``(lat, lon)``.

    The first centroid is the cell containing ``(lat, lon)`` itself;
    subsequent centroids walk outward through grid_disk rings, taking
    cells in deterministic ring-then-id order. For ``n_sub == 1``
    returns the input coordinate unchanged so the small-city path
    matches the legacy 1-endpoint-per-city behaviour exactly.
    """
    if n_sub <= 1:
        return [(lat, lon)]

    seed_cell = h3.latlng_to_cell(lat, lon, _SUB_ENDPOINT_H3_RES)
    picked: list[str] = [seed_cell]
    seen: set[str] = {seed_cell}
    ring = 1
    # grid_disk(seed, ring=R) returns cells within distance R; subtract
    # the inner disk to get just ring R. Continue outward until we have
    # enough cells. The hard cap of ring 8 keeps splits geographically
    # local even for pathologically large cities (ring 8 ≈ ~80 km at
    # res 5).
    while len(picked) < n_sub and ring <= 8:
        outer = h3.grid_disk(seed_cell, ring)
        # Sort for deterministic ordering across runs.
        new_cells = sorted(c for c in outer if c not in seen)
        for c in new_cells:
            picked.append(c)
            seen.add(c)
            if len(picked) >= n_sub:
                break
        ring += 1

    return [h3.cell_to_latlng(c) for c in picked[:n_sub]]


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
        """Generate one endpoint per *H3 cell* with Starlink users.

        The production pipeline routes at cell granularity (see
        ``domain/cell.py``), so traffic generation collapses to the
        same level: each active H3 res-5 cell is a single source
        endpoint, carrying the sum of user counts from every city
        that falls inside it.

        City-level structure is preserved as metadata on the
        :class:`CityGroup` (``city``/``country`` = dominant
        contributor, name = ``city_<ISO>_<City>[_p<i>]`` from the
        dominant sub-endpoint so the existing
        ``run_1hour.py::country_of`` parser keeps working without
        changes).

        The previous implementation materialized one endpoint per
        sub-endpoint (≈2926 for the current dataset) — a 1:1.06
        oversampling of cells (2747) caused by small nearby cities
        landing in the same hex. Those duplicates now merge into a
        single endpoint per cell, dropping a redundant abstraction
        layer between the generator and the router.

        Args:
            user_scale: Multiplier on user counts. 1.0 = real
                Starlink data (~6.4M users). 0.01 = 1% sample.
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

        # Accumulator keyed by H3 cell id. Each entry records the
        # total users that landed in this cell plus every
        # (city, country, user_count, candidate_name) tuple that
        # contributed. The dominant contributor (most users) provides
        # the cell's canonical name/country; other contributors are
        # folded into the same endpoint but their label is lost (rare:
        # ~6% of cells; never happens for major metros since
        # _spread_subcells picks distinct cells per city split).
        by_cell: dict[int, dict] = {}

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
                # Big cities still fan out across multiple H3 cells
                # so aggregate demand spreads geographically. Each
                # sub-location lands in a distinct cell (by
                # construction — _spread_subcells picks unique H3
                # indices), so the per-city split survives the
                # cell-level merge below.
                n_sub = _split_factor(city_user_count)
                sub_locs = _spread_subcells(city["lat"], city["lon"], n_sub)
                base_name = f"city_{code}_{city['city'].replace(' ', '_')}"
                per_sub = city_user_count // n_sub
                leftover = city_user_count - per_sub * n_sub
                for si, (lat, lon) in enumerate(sub_locs):
                    sub_users = per_sub + (1 if si < leftover else 0)
                    if sub_users <= 0:
                        continue
                    cid_token = h3.latlng_to_cell(lat, lon, _SUB_ENDPOINT_H3_RES)
                    cid_int = h3.str_to_int(cid_token)
                    candidate_name = (
                        base_name if n_sub == 1 else f"{base_name}_p{si}"
                    )
                    entry = by_cell.setdefault(cid_int, {"users": 0, "contributors": []})
                    entry["users"] += sub_users
                    entry["contributors"].append(
                        (city["city"], code, sub_users, candidate_name)
                    )

        # Materialize one endpoint + one CityGroup per active cell.
        # Endpoint position = cell centroid so routing and traffic
        # generation both refer to the same geometric point.
        sources: list[Endpoint] = []
        groups: list[CityGroup] = []
        for cid_int, entry in by_cell.items():
            entry["contributors"].sort(key=lambda x: -x[2])
            dom_city, dom_country, _, dom_name = entry["contributors"][0]
            c_lat, c_lon = h3.cell_to_latlng(h3.int_to_str(cid_int))
            sources.append(Endpoint(
                name=dom_name, lat_deg=c_lat, lon_deg=c_lon,
            ))
            groups.append(CityGroup(
                city=dom_city,
                country=dom_country,
                lat=c_lat,
                lon=c_lon,
                user_count=entry["users"],
                terminal_names=(dom_name,),
            ))

        if destinations is None:
            raise ValueError("destinations must be provided (no hardcoded defaults)")

        return EndpointPopulation(
            tuple(sources), destinations, tuple(groups),
        )
