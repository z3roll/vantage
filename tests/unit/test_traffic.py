"""Tests for traffic generation module."""

from __future__ import annotations

from pathlib import Path

import pytest

from vantage.domain import Endpoint, FlowKey, TrafficDemand
from vantage.traffic import (
    EndpointPopulation,
    GravityGenerator,
    UniformGenerator,
)


@pytest.fixture
def population() -> EndpointPopulation:
    sources = (
        Endpoint("user_a", 40.7, -74.0),   # New York
        Endpoint("user_b", 51.5, -0.1),     # London
    )
    destinations = (
        Endpoint("google", 37.4, -122.1),   # Mountain View
        Endpoint("facebook", 37.5, -122.1),  # Menlo Park
    )
    return EndpointPopulation(sources, destinations)


# ---------------------------------------------------------------------------
# EndpointPopulation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEndpointPopulation:

    def test_sources_and_destinations(self, population: EndpointPopulation) -> None:
        assert len(population.sources) == 2
        assert len(population.destinations) == 2

    def test_from_terminal_registry(self) -> None:
        terminals_path = Path(__file__).resolve().parents[2] / "data/processed/terminals.json"
        pop = EndpointPopulation.from_terminal_registry(terminals_path)
        assert len(pop.sources) == 106
        assert len(pop.destinations) == 5  # DEFAULT_DESTINATIONS

    def test_from_terminal_registry_custom_destinations(self) -> None:
        terminals_path = Path(__file__).resolve().parents[2] / "data/processed/terminals.json"
        dests = (Endpoint("test", 0.0, 0.0),)
        pop = EndpointPopulation.from_terminal_registry(terminals_path, destinations=dests)
        assert len(pop.destinations) == 1


# ---------------------------------------------------------------------------
# UniformGenerator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUniformGenerator:

    def test_generates_all_pairs(self, population: EndpointPopulation) -> None:
        gen = UniformGenerator(population, demand_per_flow_gbps=0.01)
        demand = gen.generate(epoch=0)
        # 2 sources × 2 destinations = 4 flows
        assert len(demand.flows) == 4

    def test_epoch_propagated(self, population: EndpointPopulation) -> None:
        gen = UniformGenerator(population)
        demand = gen.generate(epoch=5)
        assert demand.epoch == 5

    def test_uniform_demand(self, population: EndpointPopulation) -> None:
        gen = UniformGenerator(population, demand_per_flow_gbps=0.05)
        demand = gen.generate(epoch=0)
        for gbps in demand.flows.values():
            assert gbps == 0.05

    def test_flow_keys_correct(self, population: EndpointPopulation) -> None:
        gen = UniformGenerator(population)
        demand = gen.generate(epoch=0)
        expected_keys = {
            FlowKey("user_a", "google"),
            FlowKey("user_a", "facebook"),
            FlowKey("user_b", "google"),
            FlowKey("user_b", "facebook"),
        }
        assert set(demand.flows.keys()) == expected_keys

    def test_demand_is_frozen(self, population: EndpointPopulation) -> None:
        gen = UniformGenerator(population)
        demand = gen.generate(epoch=0)
        with pytest.raises(AttributeError):
            demand.epoch = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GravityGenerator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGravityGenerator:

    def test_generates_all_pairs(self, population: EndpointPopulation) -> None:
        gen = GravityGenerator(population, total_gbps=10.0)
        demand = gen.generate(epoch=0)
        assert len(demand.flows) == 4

    def test_total_demand_matches(self, population: EndpointPopulation) -> None:
        gen = GravityGenerator(population, total_gbps=10.0)
        demand = gen.generate(epoch=0)
        total = sum(demand.flows.values())
        assert abs(total - 10.0) < 1e-6

    def test_closer_gets_more(self, population: EndpointPopulation) -> None:
        """User in New York is closer to US destinations than London user."""
        gen = GravityGenerator(population, total_gbps=10.0)
        demand = gen.generate(epoch=0)
        # NY→Google vs London→Google
        ny_google = demand.flows[FlowKey("user_a", "google")]
        london_google = demand.flows[FlowKey("user_b", "google")]
        assert ny_google > london_google

    def test_deterministic(self, population: EndpointPopulation) -> None:
        gen1 = GravityGenerator(population, total_gbps=10.0)
        gen2 = GravityGenerator(population, total_gbps=10.0)
        d1 = gen1.generate(epoch=0)
        d2 = gen2.generate(epoch=0)
        for k in d1.flows:
            assert d1.flows[k] == d2.flows[k]

    def test_all_positive(self, population: EndpointPopulation) -> None:
        gen = GravityGenerator(population, total_gbps=10.0)
        demand = gen.generate(epoch=0)
        for gbps in demand.flows.values():
            assert gbps > 0
