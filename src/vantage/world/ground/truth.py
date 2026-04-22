"""Per-flow epoch-varying ground-truth RTT source.

Contract after the per-flow refactor:

  * ``sample(pop, dest, epoch, flow_id)`` returns a per-flow RTT in ms.
    Two flows with different ``flow_id`` that happen to choose the
    same ``(pop, dest)`` in the same epoch get **different** samples;
    the same ``flow_id`` always sees the same value within a given
    epoch, and reproduces bit-exactly across runs with the same
    ``seed_base``.
  * The sample seed is :func:`~vantage.common.seed.mix_seed` of
    ``(seed_base, epoch, pop, dest, flow_id)``. That keeps the draws
    call-order-free (any order of ``sample()`` calls inside one
    epoch produces the same per-flow value) and independent across
    the four identity axes.
  * A single-epoch memo keyed on ``(pop, dest, flow_id)`` makes
    repeated calls cheap. The memo is cleared the moment a different
    epoch is sampled, so memory stays at ``O(n_flows_per_epoch)``.

BL and PG still see identical truth for any *shared* hypothesis
``(pop, dest, epoch, flow_id)`` — the seed is a pure function of
those four inputs. If BL and PG route the same flow to different
PoPs, they each see truth for their own ``(pop, dest)`` pair, which
is exactly the "different routing decisions on a shared ground
reality" property the outer evaluation cares about.
"""

from __future__ import annotations

import math
import random as _random
from typing import Protocol

from vantage.common.seed import mix_seed

__all__ = ["GroundPrior", "GroundTruth"]


class GroundPrior(Protocol):
    """Deterministic one-way RTT (ms) from a PoP to a destination.

    Implementations must be pure — the truth source relies on the
    prior providing a stable median so epoch/flow jitter is the only
    source of variability.
    """

    def estimate(self, pop_code: str, dest_name: str) -> float: ...


class GroundTruth:
    """Per-flow, per-epoch truth sampler.

    Parameters
    ----------
    prior:
        A deterministic :class:`GroundPrior` that supplies the
        one-way RTT median per ``(pop, dest)``. The per-flow sample
        is drawn as ``exp(gauss(log(2·median), sigma))`` so it's a
        two-way RTT in ms centred on ``2·median``.
    seed_base:
        Run-level integer (typically
        ``derive_subseed(run_seed, "ground_truth")``) that anchors
        every sample to the run.
    sigma:
        LogNormal σ on the log-scale. 0.3 ≈ ±30 % typical deviation.
    """

    __slots__ = ("_prior", "_seed_base", "_sigma", "_cur_epoch", "_samples")

    def __init__(
        self,
        prior: GroundPrior,
        seed_base: int,
        *,
        sigma: float = 0.3,
    ) -> None:
        self._prior = prior
        self._seed_base = int(seed_base)
        self._sigma = float(sigma)
        # Single-epoch memo: (pop, dest, flow_id) → sampled RTT (ms).
        # Cleared whenever an incoming call carries a different
        # epoch so memory stays at O(n_flows_per_epoch).
        self._cur_epoch: int = -1
        self._samples: dict[tuple[str, str, str], float] = {}

    @property
    def seed_base(self) -> int:
        return self._seed_base

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(
        self,
        pop_code: str,
        dest: str,
        epoch: int,
        flow_id: str,
    ) -> float:
        """Draw the ground-truth RTT (ms) for one flow at ``epoch``.

        ``flow_id`` is any string uniquely identifying the flow's
        origin — in Argus this is the ``FlowKey.src`` terminal name,
        which makes every ``(src, dst)`` flow its own realisation
        even when many flows happen to share the same ``(pop, dst)``.
        """
        if epoch != self._cur_epoch:
            self._samples.clear()
            self._cur_epoch = epoch
        key = (pop_code, dest, flow_id)
        cached = self._samples.get(key)
        if cached is not None:
            return cached
        try:
            one_way_median = float(self._prior.estimate(pop_code, dest))
        except KeyError:
            # No prior for this pair — re-raise so callers can treat
            # it as "truth unavailable" the same way they do today.
            raise
        if not math.isfinite(one_way_median) or one_way_median <= 0:
            raise ValueError(
                f"GroundTruth: prior returned non-positive median "
                f"{one_way_median} for (pop={pop_code!r}, dest={dest!r})"
            )
        median_rtt = 2.0 * one_way_median
        rng = _random.Random(
            mix_seed(self._seed_base, epoch, pop_code, dest, flow_id)
        )
        value = math.exp(rng.gauss(math.log(median_rtt), self._sigma))
        self._samples[key] = value
        return value
