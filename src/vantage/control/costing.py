"""Ground-delay cost surfaces used by control-plane planners."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from vantage.control.knowledge import GroundKnowledge
from vantage.model import PoP

__all__ = ["build_ground_cost_lookup"]


def build_ground_cost_lookup(
    ground_knowledge: GroundKnowledge,
    *,
    current_epoch: int,
    lambda_dev: float = 1.0,
    stale_per_epoch_ms: float = 0.0,
    pops: Iterable[PoP] | None = None,
    dest_names: Iterable[str] | None = None,
) -> Callable[[str, str], float | None]:
    """Return a ``(pop_code, dest) -> RTT cost`` lookup for one plan build.

    Learned entries use ``GroundKnowledge.score`` so mean, deviation, and
    staleness are composed consistently across controllers. Cache misses fall
    back to the deterministic prior estimator when one is available.
    """
    estimator = ground_knowledge.estimator

    def compute(pop_code: str, dest: str) -> float | None:
        scored = ground_knowledge.score(
            pop_code,
            dest,
            current_epoch=current_epoch,
            lambda_dev=lambda_dev,
            stale_per_epoch_ms=stale_per_epoch_ms,
        )
        if scored is not None:
            return scored
        if estimator is None:
            return None
        try:
            return estimator.estimate(pop_code, dest) * 2
        except KeyError:
            return None

    if pops is None or dest_names is None:
        return compute

    table: dict[tuple[str, str], float | None] = {}
    dest_tuple = tuple(dest_names)
    for pop in pops:
        pop_code = pop.code
        for dest in dest_tuple:
            table[(pop_code, dest)] = compute(pop_code, dest)

    def lookup(pop_code: str, dest: str) -> float | None:
        return table.get((pop_code, dest))

    return lookup
