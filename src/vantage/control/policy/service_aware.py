"""Service-aware controller: joint sat + profiled service-class ground cost."""

from __future__ import annotations

from datetime import datetime
from types import MappingProxyType

from vantage.common.time import resolve_local_time
from vantage.control.policy.common.sat_cost import precompute_sat_cost
from vantage.domain import CostTables, NetworkSnapshot
from vantage.world.ground import GroundKnowledge
from vantage.world.ground.profiled_delay import ServiceGroundDelay


class ServiceAwareController:
    """Precompute per-epoch ground costs for service-class destinations."""

    def __init__(
        self,
        service_classes: tuple[str, ...],
        service_ground_delay: ServiceGroundDelay,
        ground_knowledge: GroundKnowledge | None = None,
        simulation_start_utc: datetime | None = None,
        epoch_interval_s: float = 3600.0,
    ) -> None:
        self._service_classes = service_classes
        self._service_delay = service_ground_delay
        self._gk = ground_knowledge or GroundKnowledge()
        self._simulation_start_utc = simulation_start_utc
        self._epoch_interval_s = epoch_interval_s

    @property
    def ground_knowledge(self) -> GroundKnowledge:
        return self._gk

    def compute_tables(self, snapshot: NetworkSnapshot) -> CostTables:
        sat_cost = precompute_sat_cost(snapshot)
        ground_cost: dict[tuple[str, str], float] = {}

        for pop in snapshot.infra.pops:
            local_hour, day_type = resolve_local_time(
                snapshot.epoch, self._epoch_interval_s,
                self._simulation_start_utc,
                self._service_delay.pop_timezones, pop.code,
            )
            for service_class in self._service_classes:
                rtt = self._gk.get_class_or_estimate(
                    pop.code,
                    service_class,
                    service_estimator=self._service_delay,
                    local_hour=local_hour,
                    day_type=day_type,
                )
                ground_cost[(pop.code, service_class)] = rtt

        return CostTables(
            epoch=snapshot.epoch,
            sat_cost=MappingProxyType(sat_cost),
            ground_cost=MappingProxyType(ground_cost),
        )
