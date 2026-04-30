"""Dashboard server and JSON persistence helpers."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DashboardWriter",
    "port_in_use",
    "start_dashboard_server",
]


def port_in_use(port: int) -> bool:
    """Whether some process is already listening on localhost:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_dashboard_server(port: int, directory: Path) -> subprocess.Popen:
    """Launch a detached ``http.server`` process rooted at ``directory``."""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(directory),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


@dataclass(slots=True)
class DashboardWriter:
    dashboard_dir: Path
    out_file: Path
    index_file: Path
    start_ts: str
    num_epochs: int
    epoch_s: float
    refresh_s: float
    user_scale: float
    max_gs_per_pop: int
    svc_names: Sequence[str]
    pop_list: Sequence[str]
    antennas_per_gs: int
    sat_feeder_cap_gbps: float
    run_seed: int
    seed_source: str
    traffic_seed: int
    ground_seed: int
    ingress_seed_base: int

    def rebuild_index(self) -> int:
        """Regenerate ``index.json`` from existing ``sim_data_*.json`` files."""
        scanned = [
            self._index_entry_from_file(path)
            for path in self.dashboard_dir.glob("sim_data_*.json")
        ]
        entries = [entry for entry in scanned if entry is not None]
        entries.sort(key=lambda entry: entry["mtime"], reverse=True)
        tmp = self.index_file.with_suffix(".json.tmp")
        with open(tmp, "w") as file:
            json.dump({"files": entries}, file)
        tmp.replace(self.index_file)
        return len(entries)

    def save_data(
        self,
        *,
        baseline: list,
        greedy: list,
        lpround: list,
        milp: list,
        latest_breakdown: dict,
        latest_pop_compare: dict,
        epoch_compare: list,
        cache_state: dict,
    ) -> None:
        out = {
            "config": {
                "num_epochs": self.num_epochs,
                "epoch_s": self.epoch_s,
                "refresh_s": self.refresh_s,
                "user_scale": self.user_scale,
                "max_gs_per_pop": self.max_gs_per_pop,
                "services": list(self.svc_names),
                "total_capacity_gbps": (
                    self.antennas_per_gs * self.sat_feeder_cap_gbps * len(self.pop_list)
                ),
                "started_at": self.start_ts,
                "run_seed": self.run_seed,
                "seed_source": self.seed_source,
                "sub_seeds": {
                    "traffic": self.traffic_seed,
                    "ground_delay": self.ground_seed,
                    "ingress": self.ingress_seed_base,
                },
                "series": ["baseline", "greedy", "lpround", "milp"],
            },
            "baseline": baseline,
            "greedy": greedy,
            "lpround": lpround,
            "milp": milp,
            "latest_breakdown": latest_breakdown,
            "latest_pop_compare": latest_pop_compare,
            "epoch_compare": epoch_compare,
            "cache_state": cache_state,
        }
        with open(self.out_file, "w") as file:
            json.dump(out, file)
        self.update_index(epochs_done=len(baseline))

    def update_index(self, epochs_done: int) -> None:
        by_name: dict[str, dict] = {}
        if self.index_file.exists():
            try:
                with open(self.index_file) as file:
                    for entry in json.load(file).get("files", []):
                        by_name[entry["filename"]] = entry
            except (OSError, json.JSONDecodeError):
                pass
        by_name[self.out_file.name] = {
            "filename": self.out_file.name,
            "timestamp": self.start_ts,
            "epochs_done": epochs_done,
            "epochs_total": self.num_epochs,
            "user_scale": self.user_scale,
            "max_gs_per_pop": self.max_gs_per_pop,
            "mtime": time.time(),
        }
        existing = {path.name for path in self.dashboard_dir.glob("sim_data_*.json")}
        entries = [
            entry for entry in by_name.values()
            if entry["filename"] in existing
        ]
        entries.sort(key=lambda entry: entry["mtime"], reverse=True)
        tmp = self.index_file.with_suffix(".json.tmp")
        with open(tmp, "w") as file:
            json.dump({"files": entries}, file)
        tmp.replace(self.index_file)

    def _index_entry_from_file(self, path: Path) -> dict | None:
        try:
            with open(path) as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        cfg = data.get("config", {})
        bl = data.get("baseline") or []
        greedy = data.get("greedy") or []
        epochs_done = max(len(bl), len(greedy))
        ts = cfg.get("started_at") or path.stem.removeprefix("sim_data_")
        return {
            "filename": path.name,
            "timestamp": ts,
            "epochs_done": epochs_done,
            "epochs_total": cfg.get("num_epochs", epochs_done),
            "user_scale": cfg.get("user_scale", 1.0),
            "max_gs_per_pop": cfg.get("max_gs_per_pop"),
            "mtime": path.stat().st_mtime,
        }
