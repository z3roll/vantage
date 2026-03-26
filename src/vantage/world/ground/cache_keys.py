"""Cache key encoding for GroundKnowledge layered cache.

Two tiers of structured keys:
- Class-level: "class:<service_class>" or
  "class:<service_class>:<day_type>:<hour>"
- Service-level: "service:<service_name>" or
  "service:<service_name>:<day_type>:<hour>" (future use)

Legacy keys (plain strings like "google") remain valid and are treated
as "legacy" tier.
"""

from __future__ import annotations

DELIMITER = ":"
CLASS_PREFIX = "class"
SERVICE_PREFIX = "service"


def encode_class_key(
    service_class: str,
    day_type: str | None = None,
    hour: int | None = None,
) -> str:
    """Encode a service-class cache key."""
    if day_type is None or hour is None:
        return f"{CLASS_PREFIX}{DELIMITER}{service_class}"
    return f"{CLASS_PREFIX}{DELIMITER}{service_class}{DELIMITER}{day_type}{DELIMITER}{hour}"


def encode_service_key(
    service_name: str,
    day_type: str | None = None,
    hour: int | None = None,
) -> str:
    """Encode a service-specific cache key."""
    if day_type is None or hour is None:
        return f"{SERVICE_PREFIX}{DELIMITER}{service_name}"
    return f"{SERVICE_PREFIX}{DELIMITER}{service_name}{DELIMITER}{day_type}{DELIMITER}{hour}"


def decode_key(key: str) -> tuple[str, str, str | None, int | None]:
    """Decode a cache key into (tier, name).

    Returns:
        ("class", "video_streaming", None, None)
        ("class", "video_streaming", "weekday", 20)
        ("service", "youtube", None, None)
        ("legacy", "google", None, None)
    """
    if DELIMITER in key:
        parts = key.split(DELIMITER)
        tier = parts[0]
        if tier in (CLASS_PREFIX, SERVICE_PREFIX):
            if len(parts) == 2:
                return tier, parts[1], None, None
            if len(parts) == 4:
                return tier, parts[1], parts[2], int(parts[3])
    return "legacy", key, None, None


def is_class_key(key: str) -> bool:
    """Check if a cache key is a class-level key."""
    return key.startswith(CLASS_PREFIX + DELIMITER)


def is_service_key(key: str) -> bool:
    """Check if a cache key is a service-level key."""
    return key.startswith(SERVICE_PREFIX + DELIMITER)
