"""Profile-based ground delay using traceroute ground-segment measurements.

Instead of geographic distance estimation, returns ground RTT based on
(pop_code, service_class) using per-PoP traceroute measurements to
google/facebook/wikipedia as anchors.

Data source: data/probe_trace/traceroute/{google,facebook,wikipedia}_summary.json
Each probe entry contains avg_ground_segment (already the ground-only RTT).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Protocol


class ServiceGroundDelay(Protocol):
    """Protocol for service-class ground delay estimation."""

    def estimate_service(
        self,
        pop_code: str,
        service_class: str,
        local_hour: int = 12,
        day_type: str = "weekday",
    ) -> float:
        """Estimate ground RTT in ms for a service class.

        Returns:
            Ground RTT in ms (round-trip).
        """
        ...


# Anchor mapping: service class -> traceroute destinations.
DEFAULT_SERVICE_ANCHOR_MAP: dict[str, tuple[str, ...]] = {
    "video_streaming": ("google",),
    "social_media": ("facebook",),
    "messaging": ("facebook",),
    "music_audio": ("google",),
    "news": ("wikipedia",),
    "generative_ai": ("google",),
    "gaming": ("google", "facebook"),
    "financial_services": ("google",),
    "ecommerce": ("google", "facebook"),
    "general_web": ("wikipedia", "google", "facebook"),
}


def load_traceroute_ground_rtt(
    traceroute_dir: str | Path,
) -> dict[str, dict[str, float]]:
    """Load per-PoP ground segment RTT from traceroute summaries.

    Reads {dest}_summary.json files, extracts avg_ground_segment per
    (pop_code, dest), averaging across probes for the same PoP.

    Returns:
        {pop_code: {dest: avg_ground_rtt_ms}}
    """
    traceroute_dir = Path(traceroute_dir)
    # Accumulate: {pop: {dest: [rtt, rtt, ...]}}
    raw: dict[str, dict[str, list[float]]] = {}

    for dest in ("google", "facebook", "wikipedia"):
        summary_path = traceroute_dir / f"{dest}_summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as f:
            data = json.load(f)
        for entry in data.values():
            if not entry or not entry.get("pop") or not entry.get("summary"):
                continue
            avg_vals = entry["summary"].get("average_values")
            if not avg_vals:
                continue
            pop_code = entry["pop"]["code"]
            gnd_rtt = avg_vals.get("avg_ground_segment")
            if gnd_rtt is None or gnd_rtt <= 0:
                continue
            raw.setdefault(pop_code, {}).setdefault(dest, []).append(gnd_rtt)

    # Average per (pop, dest)
    result: dict[str, dict[str, float]] = {}
    for pop_code, dests in raw.items():
        result[pop_code] = {}
        for dest, vals in dests.items():
            result[pop_code][dest] = sum(vals) / len(vals)
    return result


def load_pop_timezones(path: str | Path) -> dict[str, str]:
    """Load {pop_code -> IANA timezone} from pop_radar_regions.csv."""
    result: dict[str, str] = {}
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["pop_code"]] = row["timezone"]
    return result


class ProfiledGroundDelay:
    """Profile-based ground delay from traceroute ground-segment measurements.

    Lookup fallback chain:
    1. Exact: average ground RTT across anchors for (pop, service_class)
    2. Pop-level: average all dests for this pop
    3. Global: average across all pops for this service class's anchors
    4. Default constant (configurable)

    local_hour and day_type are accepted for interface stability but not
    used in v1.
    """

    def __init__(
        self,
        ground_rtt: dict[str, dict[str, float]],
        service_anchor_map: dict[str, tuple[str, ...]] | None = None,
        pop_timezones: dict[str, str] | None = None,
        default_rtt_ms: float = 20.0,
    ) -> None:
        self._ground_rtt = ground_rtt
        self._anchor_map = service_anchor_map or DEFAULT_SERVICE_ANCHOR_MAP
        self._pop_timezones = pop_timezones or {}
        self._default_rtt = default_rtt_ms

        self._cache: dict[tuple[str, str], float] = {}
        self._precompute()

    def _precompute(self) -> None:
        """Build (pop_code, service_class) -> RTT cache from anchors."""
        for pop_code, pop_data in self._ground_rtt.items():
            for svc, anchors in self._anchor_map.items():
                rtts = [pop_data[a] for a in anchors if a in pop_data]
                if rtts:
                    self._cache[(pop_code, svc)] = sum(rtts) / len(rtts)

    @property
    def pop_timezones(self) -> dict[str, str]:
        """PoP code to IANA timezone mapping."""
        return self._pop_timezones

    def estimate_service(
        self,
        pop_code: str,
        service_class: str,
        local_hour: int = 12,
        day_type: str = "weekday",
    ) -> float:
        """Estimate ground RTT in ms. See class docstring for fallback chain."""
        # Level 1: exact (pop, service_class)
        cached = self._cache.get((pop_code, service_class))
        if cached is not None:
            return cached

        # Level 2: pop-level average
        pop_data = self._ground_rtt.get(pop_code)
        if pop_data:
            all_rtts = list(pop_data.values())
            if all_rtts:
                avg = sum(all_rtts) / len(all_rtts)
                self._cache[(pop_code, service_class)] = avg
                return avg

        # Level 3: global average for this service class
        anchors = self._anchor_map.get(service_class, ())
        global_rtts: list[float] = []
        for p_data in self._ground_rtt.values():
            for a in anchors:
                if a in p_data:
                    global_rtts.append(p_data[a])
        if global_rtts:
            avg = sum(global_rtts) / len(global_rtts)
            self._cache[(pop_code, service_class)] = avg
            return avg

        # Level 4: default
        return self._default_rtt


def create_profiled_delay(
    traceroute_dir: str | Path | None = None,
    pop_regions_path: str | Path | None = None,
    default_rtt_ms: float = 20.0,
) -> ProfiledGroundDelay:
    """Factory: loads traceroute data and pop timezones.

    Default paths:
    - traceroute_dir: data/probe_trace/traceroute/
    - pop_regions_path: data/model_inputs/radar/pop_radar_regions.csv
    """
    project_root = Path(__file__).resolve().parents[4]
    if traceroute_dir is None:
        traceroute_dir = project_root / "data" / "probe_trace" / "traceroute"
    if pop_regions_path is None:
        pop_regions_path = (
            project_root / "data" / "model_inputs" / "radar" / "pop_radar_regions.csv"
        )

    ground_rtt = load_traceroute_ground_rtt(traceroute_dir)
    pop_tz = load_pop_timezones(pop_regions_path)

    return ProfiledGroundDelay(
        ground_rtt=ground_rtt,
        pop_timezones=pop_tz,
        default_rtt_ms=default_rtt_ms,
    )
