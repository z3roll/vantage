"""Command-line entrypoint for the Argus simulation."""

from __future__ import annotations

from vantage.sim.config import SimConfig, parse_args
from vantage.sim.runner import run_simulation

__all__ = ["main"]


def main() -> None:
    args = parse_args()
    config = SimConfig.from_args(args)
    run_simulation(config)
