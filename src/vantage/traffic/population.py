"""Endpoint population: source terminals and destinations."""

from __future__ import annotations

import json
from pathlib import Path

from vantage.domain import Endpoint

# Well-known CDN/server locations — global distribution
DEFAULT_DESTINATIONS: tuple[Endpoint, ...] = (
    # North America
    Endpoint("google", 37.4220, -122.0841),            # Mountain View, CA
    Endpoint("facebook", 37.4845, -122.1477),           # Menlo Park, CA
    Endpoint("amazon", 47.6062, -122.3321),             # Seattle, WA
    Endpoint("cloudflare", 37.7749, -122.4194),          # San Francisco, CA
    Endpoint("microsoft", 47.6400, -122.1300),           # Redmond, WA
    Endpoint("netflix", 37.2600, -121.9600),             # Los Gatos, CA
    Endpoint("wikipedia", 38.8951, -77.0364),            # Ashburn, VA
    Endpoint("github", 37.7749, -122.4194),              # San Francisco, CA
    Endpoint("twitter", 37.7749, -122.4194),             # San Francisco, CA
    Endpoint("apple", 37.3349, -122.0090),               # Cupertino, CA
    Endpoint("oracle", 37.5294, -122.2662),              # Redwood City, CA
    Endpoint("ibm", 41.1044, -73.7208),                  # Armonk, NY
    Endpoint("akamai_us", 42.3601, -71.0589),            # Boston, MA
    Endpoint("digitalocean", 40.7128, -74.0060),         # New York, NY
    Endpoint("att", 32.7767, -96.7970),                  # Dallas, TX
    # Europe
    Endpoint("spotify", 59.3293, 18.0686),               # Stockholm, SE
    Endpoint("sap", 49.2946, 8.6432),                    # Walldorf, DE
    Endpoint("ovh", 50.6292, 3.0573),                    # Roubaix, FR
    Endpoint("hetzner", 50.4779, 12.3375),               # Falkenstein, DE
    Endpoint("bbc", 51.5074, -0.1278),                   # London, UK
    Endpoint("yandex", 55.7558, 37.6173),                # Moscow, RU
    Endpoint("akamai_eu", 50.1109, 8.6821),              # Frankfurt, DE
    Endpoint("scaleway", 48.8566, 2.3522),               # Paris, FR
    # Asia
    Endpoint("alibaba", 30.2741, 120.1551),              # Hangzhou, CN
    Endpoint("tencent", 22.5431, 114.0579),              # Shenzhen, CN
    Endpoint("samsung", 37.3595, 127.1052),              # Suwon, KR
    Endpoint("sony", 35.6762, 139.6503),                 # Tokyo, JP
    Endpoint("rakuten", 35.6295, 139.7436),              # Tokyo, JP
    Endpoint("naver", 37.3861, 127.1152),                # Seongnam, KR
    Endpoint("baidu", 39.9856, 116.3077),                # Beijing, CN
    Endpoint("akamai_asia", 1.3521, 103.8198),           # Singapore
    # South America
    Endpoint("mercadolibre", -34.6037, -58.3816),        # Buenos Aires, AR
    Endpoint("uol", -23.5505, -46.6333),                 # São Paulo, BR
    # Oceania
    Endpoint("atlassian", -33.8688, 151.2093),           # Sydney, AU
    # Africa
    Endpoint("jumia", 6.5244, 3.3792),                   # Lagos, NG
    # Middle East
    Endpoint("souq", 25.2048, 55.2708),                  # Dubai, AE
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
