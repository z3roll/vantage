"""Simulation CLI and path configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from vantage.common.seed import derive_subseed, fresh_run_seed

__all__ = [
    "CELL_CACHE",
    "DASHBOARD_DIR",
    "DATA_DIR",
    "EPOCH_S",
    "LAND_GEOJSON",
    "N_ANTENNAS_PER_GS",
    "REFRESH",
    "REPO_ROOT",
    "SAT_FEEDER_CAP_GBPS",
    "XML",
    "SeedBundle",
    "SimConfig",
    "parse_args",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "src" / "vantage" / "config"
XML = DATA_DIR / "Starlink.xml"
LAND_GEOJSON = DATA_DIR / "ne_countries.geojson"
CELL_CACHE = REPO_ROOT / "data" / "processed" / "land_cells_res5.json"
DASHBOARD_DIR = REPO_ROOT / "dashboard"

EPOCH_S = 1.0
REFRESH = 15
N_ANTENNAS_PER_GS = 8
SAT_FEEDER_CAP_GBPS = 20.0


@dataclass(frozen=True, slots=True)
class SeedBundle:
    run_seed: int
    traffic_seed: int
    ground_seed: int
    ingress_seed_base: int
    seed_source: str

    @classmethod
    def from_cli_seed(cls, cli_seed: int | None) -> SeedBundle:
        run_seed = cli_seed if cli_seed is not None else fresh_run_seed()
        return cls(
            run_seed=run_seed,
            traffic_seed=derive_subseed(run_seed, "traffic"),
            ground_seed=derive_subseed(run_seed, "ground_delay"),
            ingress_seed_base=derive_subseed(run_seed, "ingress"),
            seed_source="cli" if cli_seed is not None else "auto",
        )


@dataclass(frozen=True, slots=True)
class SimConfig:
    num_epochs: int
    user_scale: float
    port: int
    no_browser: bool
    no_serve: bool
    max_gs_per_pop: int
    seeds: SeedBundle

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SimConfig:
        return cls(
            num_epochs=args.epochs,
            user_scale=args.user_scale,
            port=args.port,
            no_browser=args.no_browser,
            no_serve=args.no_serve,
            max_gs_per_pop=args.max_gs_per_pop,
            seeds=SeedBundle.from_cli_seed(args.seed),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vantage end-to-end simulation with auto-launched live dashboard."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of 1-second epochs to simulate (default: 60)",
    )
    parser.add_argument(
        "--user-scale",
        type=float,
        default=5.0,
        help="Starlink user-count multiplier (default: 5.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port for the dashboard (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open the browser (dashboard server still runs)",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Don't start the dashboard HTTP server at all (benchmark mode)",
    )
    parser.add_argument(
        "--max-gs-per-pop",
        type=int,
        default=0,
        help=(
            "Experimental: cap each PoP at N attached GSs (0 = no cap). "
            "Keeps the N closest by backhaul delay so popular PoPs hit "
            "capacity earlier, exposing PG / DP differences under pressure."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Run-level seed controlling every stochastic subsystem "
            "(traffic AR(1)/Poisson, ground-delay LogNormal sampling, "
            "ingress-sat selection). If omitted, a fresh random seed is drawn."
        ),
    )
    return parser.parse_args()
