"""Ground delay estimation — measurement-only.

Historical note: this module used to expose ``HaversineDelay`` (great
circle × fiber speed) and ``FiberGraphDelay`` (Dijkstra over the ITU
submarine-cable graph). Both have been **deleted** because they
fabricated ground latencies for arbitrary ``(pop, destination)``
pairs based on geography, and no amount of "detour factor" heuristics
can reproduce real-world peering and BGP effects. We now require
every ground delay lookup to be backed by an actual measurement.

The remaining surface is deliberately small:

    * :class:`GroundDelay` — Protocol. Takes a ``(pop_code, dest_name)``
      pair, returns a one-way ground RTT in ms. Raises :class:`KeyError`
      on unknown pairs — there is **no fallback**, by design.
    * :class:`MeasuredGroundDelay` — the only concrete implementation.
      Backed by a frozen mapping loaded from traceroute summaries
      under ``data/probe_trace/traceroute/``. Currently covers
      ``29 PoPs × 3 services (google / facebook / wikipedia) = 87``
      measured pairs.

Any code path that previously relied on a geographic estimator now
either receives a concrete measurement or must handle :class:`KeyError`
explicitly — the data plane treats unknown pairs as *unrouted flows*,
not silently-zero delays.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

__all__ = [
    "DEFAULT_MEASURED_SERVICES",
    "GroundDelay",
    "MeasuredGroundDelay",
]


DEFAULT_MEASURED_SERVICES: tuple[str, ...] = ("google", "facebook", "wikipedia")
"""The three destinations we currently have ground-truth traceroute
data for. Any attempt to route a flow to a destination outside this
set will raise :class:`KeyError` unless a custom measurement set is
supplied."""


class GroundDelay(Protocol):
    """Protocol for ground segment delay lookup (PoP → destination).

    Implementations must return a one-way ground RTT in ms for a
    ``(pop_code, dest_name)`` pair they have data for, and raise
    :class:`KeyError` otherwise. The Protocol intentionally does **not**
    expose a fallback hook — any fallback must be implemented at a
    higher layer (e.g. the data plane marking a flow unrouted).
    """

    def estimate(self, pop_code: str, dest_name: str) -> float:
        """Return one-way ground delay in ms for ``(pop_code, dest_name)``.

        Raises:
            KeyError: If no measurement is available for the pair.
        """
        ...


@dataclass(frozen=True, slots=True)
class MeasuredGroundDelay:
    """Ground delay backed by a fixed ``(pop, destination) → RTT`` table.

    Construct via :meth:`from_traceroute_dir` for real experiments, or
    via the direct constructor for unit tests with hand-injected data.
    The internal mapping is frozen into a :class:`MappingProxyType`
    in ``__post_init__`` so the table cannot be mutated after load.

    All values are one-way RTT in milliseconds. The loader divides
    traceroute-measured ground segment RTT by 2 so the result matches
    the ``one-way`` convention used by :class:`GroundDelay`.
    """

    one_way_rtt_ms: Mapping[tuple[str, str], float]

    def __post_init__(self) -> None:
        if not isinstance(self.one_way_rtt_ms, MappingProxyType):
            object.__setattr__(
                self, "one_way_rtt_ms", MappingProxyType(dict(self.one_way_rtt_ms))
            )

    # --- Protocol -----------------------------------------------------------

    def estimate(self, pop_code: str, dest_name: str) -> float:
        """Return one-way ground delay (ms) or raise :class:`KeyError`."""
        try:
            return self.one_way_rtt_ms[(pop_code, dest_name)]
        except KeyError:
            raise KeyError(
                f"no measured ground RTT for pair (pop={pop_code!r}, "
                f"dest={dest_name!r}); MeasuredGroundDelay is strict — "
                f"there is no geographic fallback"
            ) from None

    # --- Query helpers ------------------------------------------------------

    def has(self, pop_code: str, dest_name: str) -> bool:
        """Whether ``(pop_code, dest_name)`` has a measurement."""
        return (pop_code, dest_name) in self.one_way_rtt_ms

    def pops(self) -> frozenset[str]:
        """The set of PoPs with at least one measured destination."""
        return frozenset(pop for (pop, _) in self.one_way_rtt_ms)

    def destinations(self) -> frozenset[str]:
        """The set of destinations measured from at least one PoP."""
        return frozenset(dst for (_, dst) in self.one_way_rtt_ms)

    def __len__(self) -> int:
        return len(self.one_way_rtt_ms)

    # --- Loaders ------------------------------------------------------------

    @classmethod
    def empty(cls) -> MeasuredGroundDelay:
        """Zero-entry instance. Every lookup raises.

        Use this only when you *want* a ground-delay provider present
        for type-correctness but with no data wired up yet — every
        forward pass will report 100% unrouted.
        """
        return cls(one_way_rtt_ms=MappingProxyType({}))

    @classmethod
    def from_traceroute_dir(
        cls,
        traceroute_dir: str | Path,
        services: tuple[str, ...] = DEFAULT_MEASURED_SERVICES,
        *,
        require_all_services: bool = True,
    ) -> MeasuredGroundDelay:
        """Build a :class:`MeasuredGroundDelay` from per-service summary files.

        Reads ``{traceroute_dir}/{service}_summary.json`` for each
        entry in ``services``. Each file is expected to contain the
        schema produced by the preprocessing pipeline: a dict of
        ``probe_id → {probe_*, pop, summary: {average_values:
        {avg_ground_segment: float, ...}}}``. For every entry with a
        non-null ``pop.code`` and a positive ``avg_ground_segment``,
        we accumulate one-way RTT samples into a
        ``(pop_code, service) → [rtt, ...]`` bucket and then take the
        mean.

        Traceroute samples are **round-trip** in the source file, so
        the loader divides by 2 before storing. Callers can therefore
        treat :meth:`estimate` as "one-way" uniformly, and the forward
        layer can double it when it needs RTT.

        Args:
            traceroute_dir: Directory containing the summary files.
            services: Which services to load. Defaults to
                :data:`DEFAULT_MEASURED_SERVICES`.
            require_all_services: When ``True`` (the default), every
                requested service file must exist. Missing files raise
                :class:`FileNotFoundError` so production experiments
                can't silently degrade to a partial table. Set to
                ``False`` for tooling or tests that intentionally load
                a subset.

        Raises:
            FileNotFoundError: If ``traceroute_dir`` itself does not
                exist, if ``require_all_services=True`` and any
                requested file is missing, or if every requested
                file is missing.
            ValueError: If the loaded set is empty (all samples
                invalid after filtering).
        """
        traceroute_dir = Path(traceroute_dir)
        if not traceroute_dir.exists():
            raise FileNotFoundError(
                f"traceroute directory not found: {traceroute_dir}"
            )

        # Accumulate raw round-trip samples per (pop, service).
        raw: dict[tuple[str, str], list[float]] = {}
        missing_files: list[str] = []

        for service in services:
            summary_path = traceroute_dir / f"{service}_summary.json"
            if not summary_path.exists():
                missing_files.append(service)
                continue
            with summary_path.open() as f:
                data = json.load(f)
            for entry in data.values():
                if not entry or not entry.get("pop") or not entry.get("summary"):
                    continue
                avg_vals = entry["summary"].get("average_values")
                if not avg_vals:
                    continue
                pop_code = entry["pop"].get("code")
                if not pop_code:
                    continue
                round_trip_ms = avg_vals.get("avg_ground_segment")
                if round_trip_ms is None or round_trip_ms <= 0:
                    continue
                raw.setdefault((pop_code, service), []).append(float(round_trip_ms))

        if missing_files == list(services):
            raise FileNotFoundError(
                f"no traceroute summary files found in {traceroute_dir}; "
                f"looked for {', '.join(f'{s}_summary.json' for s in services)}"
            )
        if require_all_services and missing_files:
            raise FileNotFoundError(
                f"require_all_services=True but some service summary files are "
                f"missing in {traceroute_dir}: "
                f"{', '.join(f'{s}_summary.json' for s in missing_files)}. "
                f"Pass require_all_services=False to accept partial coverage."
            )

        # Take the mean across probes for each (pop, service) and
        # convert round-trip → one-way. Note: :mod:`profiled_delay`
        # intentionally keeps round-trip values (it serves a
        # different Protocol, ``ServiceGroundDelay``), so the two
        # loaders that read ``avg_ground_segment`` from the same JSON
        # files pin *different* conventions. This division by 2 is
        # the contract for :class:`MeasuredGroundDelay` only.
        one_way: dict[tuple[str, str], float] = {
            key: sum(samples) / len(samples) / 2.0
            for key, samples in raw.items()
        }

        if not one_way:
            raise ValueError(
                f"traceroute directory {traceroute_dir} produced no valid "
                "(pop, service) samples — every entry was missing a pop.code "
                "or a positive avg_ground_segment"
            )

        return cls(one_way_rtt_ms=MappingProxyType(one_way))
