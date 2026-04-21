"""Traffic generator: analytical per-city demand aggregation.

:class:`FlowLevelGenerator` is the single generator in use: Gravity
spatial distribution, Bounded Pareto flow sizes, Poisson arrivals
with diurnal modulation, per-flow lifecycle.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from vantage.common import haversine_km
from vantage.domain import FlowKey, TrafficDemand
from vantage.traffic.population import EndpointPopulation


@dataclass(frozen=True, slots=True)
class _UserProfile:
    """Parsed user profile from config."""

    name: str
    fraction: float
    r: float
    alpha: float
    size_min: float
    size_max: float
    rate_fast_prob: float
    rate_slow_log_mean: float
    rate_slow_log_std: float
    rate_fast_log_mean: float
    rate_fast_log_std: float
    mean_size_bits: float = 0.0   # pre-computed E[size] in bits
    mean_rate_gbps: float = 0.0   # pre-computed E[rate] in Gbps


class FlowLevelGenerator:
    """Analytical per-city traffic generator.

    Instead of tracking individual flows, computes aggregate demand
    per ``(city, destination)`` pair analytically each epoch:

        demand(c, d) = N_c × frac × active(local_hour) × r
                       × E[size] × gravity(c, d) × burst

    **User types**: heavy/medium/light with different ``r``, size, and
    rate distributions (from ``config/user_profiles.json``).

    **Diurnal**: per-city based on local timezone (from longitude).

    **Burstiness**: AR(1) noise per city (temporal correlation).

    **Gravity**: per-city destination weights (distance to nearest node).

    O(cities × profiles × dests) ≈ O(64K) per epoch, no flow objects.
    """

    def __init__(
        self,
        population: EndpointPopulation,
        config_dir: str | Path,
        epoch_interval_s: float = 1.0,
        *,
        dst_weights: dict[str, float] | None = None,
        dst_locations: dict[str, list[dict]] | None = None,
        bottleneck_gbps: float = 0.05,
        ar1_phi: float = 0.95,
        ar1_sigma: float = 0.3,
        seed: int = 42,
    ) -> None:
        import json
        from pathlib import Path as _Path

        self._dt = epoch_interval_s
        self._bneck = bottleneck_gbps
        self._population = population
        self._dst_weights_raw = dst_weights or {}
        self._dst_locations = dst_locations or {}
        self._rng = _random.Random(seed)

        config_dir = _Path(config_dir)

        # Load user profiles and pre-compute E[size], E[rate]
        with (config_dir / "user_profiles.json").open() as f:
            raw_profiles = json.load(f)
        self._profiles: list[_UserProfile] = []
        for name, p in raw_profiles.items():
            if "fraction" not in p:
                continue
            mean_size = _bounded_pareto_mean(
                p["pareto_alpha"], p["flow_size_min_bytes"], p["flow_size_max_bytes"],
            )
            mean_rate = _bimodal_mean_rate(
                p["rate_fast_prob"],
                p["rate_slow_log_mean"], p["rate_slow_log_std"],
                p["rate_fast_log_mean"], p["rate_fast_log_std"],
                bottleneck_gbps,
            )
            self._profiles.append(_UserProfile(
                name=name,
                fraction=p["fraction"],
                r=p["flow_rate_per_user"],
                alpha=p["pareto_alpha"],
                size_min=p["flow_size_min_bytes"],
                size_max=p["flow_size_max_bytes"],
                rate_fast_prob=p["rate_fast_prob"],
                rate_slow_log_mean=p["rate_slow_log_mean"],
                rate_slow_log_std=p["rate_slow_log_std"],
                rate_fast_log_mean=p["rate_fast_log_mean"],
                rate_fast_log_std=p["rate_fast_log_std"],
                mean_size_bits=mean_size * 8,
                mean_rate_gbps=mean_rate,
            ))

        # Load diurnal curve (24 hourly values)
        with (config_dir / "diurnal_curve.json").open() as f:
            self._diurnal_curve: list[float] = json.load(f)["curve"]

        # Per-city state
        self._city_groups = population.city_groups
        self._city_utc_offsets: list[float] = [
            round(cg.lon / 15.0) for cg in self._city_groups
        ]

        # AR(1) noise state per city
        self._ar1_phi = ar1_phi
        self._ar1_sigma = ar1_sigma
        self._ar1_state: list[float] = [0.0] * len(self._city_groups)

        # Per-city destination gravity weights (normalized, not CDF)
        self._city_dst_keys: list[list[str]] = []
        self._city_dst_weights: list[list[float]] = []
        self._build_city_gravity()

    # --- Per-city gravity -------------------------------------------------

    def _build_city_gravity(self) -> None:
        dsts = self._population.destinations
        for ci, cg in enumerate(self._city_groups):
            keys: list[str] = []
            raw: list[float] = []
            for dst in dsts:
                dw = self._dst_weights_raw.get(dst.name, 1.0)
                locs = self._dst_locations.get(dst.name)
                if locs:
                    dist = min(
                        haversine_km(cg.lat, cg.lon, loc["lat"], loc["lon"])
                        for loc in locs
                    )
                else:
                    dist = haversine_km(cg.lat, cg.lon, dst.lat_deg, dst.lon_deg)
                keys.append(dst.name)
                raw.append(dw / max(dist, 100.0))

            total = sum(raw)
            weights = [w / total for w in raw] if total > 0 else [1.0 / len(raw)] * len(raw)
            self._city_dst_keys.append(keys)
            self._city_dst_weights.append(weights)

    # --- Diurnal (per-city local time) ------------------------------------

    def _active_fraction(self, epoch: int, city_idx: int) -> float:
        utc_hour = (epoch * self._dt / 3600.0) % 24.0
        local_hour = (utc_hour + self._city_utc_offsets[city_idx]) % 24.0
        h0 = int(local_hour) % 24
        h1 = (h0 + 1) % 24
        frac = local_hour - int(local_hour)
        return self._diurnal_curve[h0] * (1 - frac) + self._diurnal_curve[h1] * frac

    # --- AR(1) burstiness -------------------------------------------------

    def _ar1_factor(self, city_idx: int) -> float:
        eps = self._rng.gauss(0, self._ar1_sigma)
        self._ar1_state[city_idx] = (
            self._ar1_phi * self._ar1_state[city_idx] + eps
        )
        return math.exp(self._ar1_state[city_idx])

    # --- generate() -------------------------------------------------------

    def generate(self, epoch: int) -> TrafficDemand:
        agg: dict[FlowKey, float] = {}
        dst_keys_all = self._city_dst_keys
        dst_wts_all = self._city_dst_weights

        for ci, cg in enumerate(self._city_groups):
            active_frac = self._active_fraction(epoch, ci)
            burst = self._ar1_factor(ci)
            src = cg.terminal_names[0]
            dst_keys = dst_keys_all[ci]
            dst_wts = dst_wts_all[ci]

            for profile in self._profiles:
                # Instantaneous concurrent demand (independent of dt):
                #   arrival_rate = N × frac × active × r  (flows/sec)
                #   E[duration]  = E[size] / E[rate]       (seconds)
                #   concurrent   = arrival_rate × E[duration]  (Little's Law)
                #   demand_gbps  = concurrent × E[rate]
                #                = arrival_rate × E[size_bits] / 1e9
                #
                # Poisson noise: sample the number of concurrent flows
                # from Poisson(concurrent). When concurrent < 1 (e.g.
                # small city at night), this naturally produces 0 most
                # of the time.
                arrival_rate = (cg.user_count * profile.fraction
                                * active_frac * profile.r * burst)
                mean_duration = (profile.mean_size_bits / 1e9) / max(profile.mean_rate_gbps, 1e-12)
                concurrent = arrival_rate * mean_duration
                n_concurrent = _poisson(self._rng, concurrent)
                if n_concurrent == 0:
                    continue

                demand_gbps = n_concurrent * profile.mean_rate_gbps

                # Distribute across destinations by gravity
                for di in range(len(dst_keys)):
                    d = demand_gbps * dst_wts[di]
                    if d > 1e-9:
                        key = FlowKey(src, dst_keys[di])
                        agg[key] = agg.get(key, 0.0) + d

        return TrafficDemand(epoch=epoch, flows=MappingProxyType(agg))


def _bounded_pareto_mean(alpha: float, lo: float, hi: float) -> float:
    """E[X] for Bounded Pareto(α, lo, hi)."""
    if abs(alpha - 1.0) < 1e-9:
        return lo * hi * math.log(hi / lo) / (hi - lo)
    num = alpha * lo ** alpha * (1.0 / lo ** (alpha - 1) - 1.0 / hi ** (alpha - 1))
    den = (alpha - 1.0) * (1.0 - (lo / hi) ** alpha)
    return num / den if den != 0 else (lo + hi) / 2


def _bimodal_mean_rate(
    p_fast: float,
    slow_mu: float, slow_sigma: float,
    fast_mu: float, fast_sigma: float,
    bottleneck_gbps: float,
) -> float:
    """E[rate] in Gbps for the bimodal LogNormal mixture."""
    # E[exp(N(mu, sigma^2))] = exp(mu + sigma^2/2)
    e_slow = math.exp(slow_mu + slow_sigma ** 2 / 2)  # Mbps
    e_fast = math.exp(fast_mu + fast_sigma ** 2 / 2)   # Mbps
    e_rate_mbps = (1 - p_fast) * e_slow + p_fast * e_fast
    return min(e_rate_mbps, bottleneck_gbps * 1000) / 1000  # → Gbps


def _poisson(rng: _random.Random, lam: float) -> int:
    """Sample from Poisson(λ) using Knuth's algorithm."""
    if lam <= 0:
        return 0
    if lam > 500:
        # Normal approximation for large λ.
        return max(0, int(rng.gauss(lam, math.sqrt(lam)) + 0.5))
    exp_neg_lam = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p < exp_neg_lam:
            return k
        k += 1
