"""Control subsystem: TE controller factory and feedback protocol."""

from vantage.control.controller import (
    SupportsGroundFeedback,
    create_controller,
)

__all__ = [
    "SupportsGroundFeedback",
    "create_controller",
]
