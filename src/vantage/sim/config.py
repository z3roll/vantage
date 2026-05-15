"""Simulation CLI and path configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from vantage.common.seed import derive_subseed, fresh_run_seed

__all__ = [
    "CELL_CACHE",
    "CONTROL_ALGORITHMS",
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
    "normalize_control_algorithms",
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
CONTROL_ALGORITHMS: tuple[str, ...] = (
    "baseline",
    "optimizer_baseline",
    "progressive",
    "optimizer",
    "greedy",
    "lpround",
    "milp",
)
_CONTROL_ALIASES = {
    "all": "all",
    "bl": "baseline",
    "nearest": "baseline",
    "nearest_pop": "baseline",
    "opt_bl": "optimizer_baseline",
    "optbaseline": "optimizer_baseline",
    "optimizerbaseline": "optimizer_baseline",
    "path_baseline": "optimizer_baseline",
    "standalone_baseline": "optimizer_baseline",
    "prog": "progressive",
    "progressive_spillover": "progressive",
    "opt": "optimizer",
    "pathaware": "optimizer",
    "path_aware": "optimizer",
    "lp": "lpround",
    "mip": "milp",
}


def normalize_control_algorithms(values: list[str] | None) -> tuple[str, ...]:
    """Normalize CLI control-policy selection."""
    if not values:
        return CONTROL_ALGORITHMS
    requested: list[str] = []
    for raw_value in values:
        for token in raw_value.split(","):
            name = token.strip().lower().replace("-", "_")
            if not name:
                continue
            name = _CONTROL_ALIASES.get(name, name)
            if name == "all":
                return CONTROL_ALGORITHMS
            if name not in CONTROL_ALGORITHMS:
                choices = ", ".join(("all",) + CONTROL_ALGORITHMS)
                raise ValueError(f"unknown control algorithm {token!r}; choose from {choices}")
            if name not in requested:
                requested.append(name)
    if not requested:
        return CONTROL_ALGORITHMS
    return tuple(requested)


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
    control_algorithms: tuple[str, ...]
    egress_top_k: int
    port: int
    no_browser: bool
    no_serve: bool
    max_gs_per_pop: int
    seeds: SeedBundle
    enforce_isl_capacity: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SimConfig:
        return cls(
            num_epochs=args.epochs,
            user_scale=args.user_scale,
            control_algorithms=normalize_control_algorithms(args.control_algorithms),
            egress_top_k=args.egress_top_k,
            port=args.port,
            no_browser=args.no_browser,
            no_serve=args.no_serve,
            max_gs_per_pop=args.max_gs_per_pop,
            seeds=SeedBundle.from_cli_seed(args.seed),
            enforce_isl_capacity=not args.disable_isl_capacity,
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
        "--control",
        "--controls",
        dest="control_algorithms",
        action="append",
        default=None,
        help=(
            "Control algorithm(s) to run: all, baseline, optimizer_baseline, "
            "progressive, greedy, optimizer, lpround, milp. Comma-separated "
            "and repeated forms are both accepted (default: all)."
        ),
    )
    parser.add_argument(
        "--egress-top-k",
        "--gateway-top-k",
        "--egress-sats-per-gs",
        dest="egress_top_k",
        type=int,
        default=N_ANTENNAS_PER_GS,
        choices=range(1, N_ANTENNAS_PER_GS + 1),
        metavar=f"1..{N_ANTENNAS_PER_GS}",
        help=(
            "Number of highest-elevation visible egress satellites retained "
            f"per GS for forward candidates (default: {N_ANTENNAS_PER_GS})."
        ),
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
            "capacity earlier, exposing Greedy / DP differences under pressure."
        ),
    )
    parser.add_argument(
        "--disable-isl-capacity",
        action="store_true",
        help=(
            "Experiment: ignore ISL capacity in forward path selection and "
            "measurement; sat feeder and GS feeder limits still apply."
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
