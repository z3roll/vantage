"""Time resolution utilities for epoch → local time conversion."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def resolve_local_time(
    epoch: int,
    epoch_interval_s: float,
    simulation_start_utc: datetime | None,
    pop_timezones: dict[str, str],
    pop_code: str,
) -> tuple[int, str]:
    """Convert simulation epoch to (local_hour, day_type) for a PoP.

    Args:
        epoch: Current simulation epoch index.
        epoch_interval_s: Duration of each epoch in seconds.
        simulation_start_utc: Absolute UTC time of epoch 0.
        pop_timezones: Mapping of pop_code → IANA timezone string.
        pop_code: PoP code to look up timezone.

    Returns:
        Tuple of (local_hour 0-23, day_type "weekday"|"weekend").
    """
    if simulation_start_utc is None:
        return 12, "weekday"

    utc_time = simulation_start_utc + timedelta(seconds=epoch * epoch_interval_s)
    tz_str = pop_timezones.get(pop_code, "UTC")
    local_time = utc_time.astimezone(ZoneInfo(tz_str))
    return local_time.hour, "weekday" if local_time.weekday() < 5 else "weekend"
