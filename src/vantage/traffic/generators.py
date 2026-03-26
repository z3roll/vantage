"""Traffic generators: produce demand per epoch."""

from __future__ import annotations

from types import MappingProxyType

from vantage.common import haversine_km
from vantage.domain import FlowKey, TrafficDemand
from vantage.traffic.population import EndpointPopulation


class UniformGenerator:
    """Generates uniform traffic demand across all src-dst pairs.

    Every source sends the same demand to every destination.
    """

    def __init__(
        self,
        population: EndpointPopulation,
        demand_per_flow_gbps: float = 0.01,
    ) -> None:
        self._population = population
        self._demand = demand_per_flow_gbps

    def generate(self, epoch: int) -> TrafficDemand:
        flows: dict[FlowKey, float] = {}
        for src in self._population.sources:
            for dst in self._population.destinations:
                flows[FlowKey(src=src.name, dst=dst.name)] = self._demand
        return TrafficDemand(epoch=epoch, flows=MappingProxyType(flows))


class RealisticGenerator:
    """Generates realistic non-uniform traffic that evolves over epochs.

    Each terminal accesses a subset of destinations. New destinations
    appear gradually over time, simulating real user behavior where
    traffic patterns shift and new services emerge.

    - Each terminal starts with 2-4 destinations
    - Every few epochs, some terminals discover new destinations
    - Popularity follows Zipf distribution (few very popular, long tail)
    """

    def __init__(
        self,
        population: EndpointPopulation,
        demand_per_flow_gbps: float = 0.01,
        initial_dests_per_terminal: int = 3,
        new_dest_probability: float = 0.15,
        seed: int = 42,
    ) -> None:
        self._population = population
        self._demand = demand_per_flow_gbps
        self._rng = __import__("random").Random(seed)

        sources = population.sources
        dests = population.destinations

        # Zipf-like popularity: first destinations are more popular
        weights = [1.0 / (i + 1) for i in range(len(dests))]
        total_w = sum(weights)
        self._dest_probs = [w / total_w for w in weights]

        # Each terminal starts with a random subset of destinations
        self._terminal_dests: dict[str, set[str]] = {}
        for src in sources:
            n = min(initial_dests_per_terminal, len(dests))
            chosen = set()
            while len(chosen) < n:
                idx = self._weighted_choice()
                chosen.add(dests[idx].name)
            self._terminal_dests[src.name] = chosen

        self._new_dest_prob = new_dest_probability

    def _weighted_choice(self) -> int:
        r = self._rng.random()
        cumulative = 0.0
        for i, p in enumerate(self._dest_probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(self._dest_probs) - 1

    def generate(self, epoch: int) -> TrafficDemand:
        dests = self._population.destinations

        # Each epoch, some terminals discover new destinations
        if epoch > 0:
            for src in self._population.sources:
                if self._rng.random() < self._new_dest_prob:
                    idx = self._weighted_choice()
                    self._terminal_dests[src.name].add(dests[idx].name)

        flows: dict[FlowKey, float] = {}
        for src in self._population.sources:
            for dst_name in self._terminal_dests[src.name]:
                flows[FlowKey(src=src.name, dst=dst_name)] = self._demand
        return TrafficDemand(epoch=epoch, flows=MappingProxyType(flows))


class GravityGenerator:
    """Generates traffic proportional to inverse distance (gravity model).

    Demand between (src, dst) is proportional to 1/distance.
    Closer pairs get more traffic. Normalized so total demand = total_gbps.
    """

    def __init__(
        self,
        population: EndpointPopulation,
        total_gbps: float = 10.0,
    ) -> None:
        self._population = population
        self._total = total_gbps
        self._weights = self._compute_weights()

    def _compute_weights(self) -> dict[FlowKey, float]:
        raw: dict[FlowKey, float] = {}
        for src in self._population.sources:
            for dst in self._population.destinations:
                dist = haversine_km(
                    src.lat_deg, src.lon_deg, dst.lat_deg, dst.lon_deg
                )
                # Avoid division by zero; minimum distance 100 km
                raw[FlowKey(src.name, dst.name)] = 1.0 / max(dist, 100.0)

        total_weight = sum(raw.values())
        return {k: v / total_weight for k, v in raw.items()}

    def generate(self, epoch: int) -> TrafficDemand:
        flows = {k: w * self._total for k, w in self._weights.items()}
        return TrafficDemand(epoch=epoch, flows=MappingProxyType(flows))
