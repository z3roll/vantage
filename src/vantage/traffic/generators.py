"""Traffic generators: produce demand per epoch."""

from __future__ import annotations

from types import MappingProxyType

from vantage.domain import FlowKey, TrafficDemand
from vantage.traffic.population import EndpointPopulation
from vantage.common import haversine_km


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
