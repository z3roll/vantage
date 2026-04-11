"""Offline preprocessing: raw data → clean JSON for the TE system.

Run once:  uv run python -m vantage.config.preprocess

Reads messy raw data (GeoJSON, CSV, JSON) from data/raw/ and writes
clean, normalized JSON files to data/processed/.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vantage.common import C_FIBER_KM_S, haversine_km

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

SCHEMA_VERSION = 2
# v2 (2026-04-11): GroundStation.min_capacity removed; max_capacity
# reinterpreted as per-GS aggregate Ka feeder capacity in Gbps
# (= num_antennas × 10.0). See domain/capacity_view.py.


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -- Ground Stations ----------------------------------------------------------


def preprocess_ground_stations() -> list[dict[str, Any]]:
    """Raw starlink_gs.json → clean list of dicts."""
    path = RAW_DIR / "infra/gs_pop_peering/ground_staion/starlink_gs.json"
    with path.open() as f:
        raw: dict[str, dict[str, Any]] = json.load(f)

    # Per-GS feeder aggregate capacity: each Ka antenna handles ~10 Gbps,
    # so the station total is num_antennas × 10.0 Gbps. This is the
    # single physical fact source for GroundStation.max_capacity; the
    # original raw FCC "mincapacity/maxcapacity" fields were RF bandwidth
    # metadata and are intentionally discarded here.
    PER_ANTENNA_GBPS = 10.0

    stations: list[dict[str, Any]] = []
    for uuid, entry in raw.items():
        num_antennas = int(entry["numAntennas"])
        stations.append({
            "gs_id": uuid,
            "lat_deg": float(entry["lat"]),
            "lon_deg": float(entry["lng"]),
            "country": str(entry["country"]),
            "town": str(entry.get("town", "")),
            "num_antennas": num_antennas,
            "min_elevation_deg": float(entry["minElevation"]),
            "enabled": bool(entry["enabled"]),
            "uplink_ghz": float(entry.get("uplinkGhz", 0.0)),
            "downlink_ghz": float(entry.get("downlinkGhz", 0.0)),
            "max_capacity": num_antennas * PER_ANTENNA_GBPS,
            "temporary": bool(entry.get("temporary", False)),
        })

    stations.sort(key=lambda g: g["gs_id"])
    return stations


# -- PoPs ----------------------------------------------------------------------


def preprocess_pops() -> list[dict[str, Any]]:
    """Raw starlink_pop.geojson → clean list of dicts.

    Only Point features with folder="PoPs & Backbone" are extracted.
    """
    path = RAW_DIR / "infra/gs_pop_peering/pop/starlink_pop.geojson"
    with path.open() as f:
        raw = json.load(f)

    pops: list[dict[str, Any]] = []
    for feat in raw["features"]:
        if feat["geometry"]["type"] != "Point":
            continue
        props = feat["properties"]
        if props.get("folder") != "PoPs & Backbone":
            continue

        name: str = props["Name"]
        coords = feat["geometry"]["coordinates"]

        # Normalize: non-breaking space, trailing whitespace
        name_clean = name.replace("\xa0", " ").strip()
        parts = name_clean.split(" - ", maxsplit=1)
        if len(parts) == 2:
            code = parts[0].strip().split("/")[0].strip().lower()
            site_id = parts[1].strip().lower()
        else:
            code = name_clean.strip().lower()
            site_id = code

        pops.append({
            "site_id": site_id,
            "code": code,
            "name": name_clean,
            "lat_deg": float(coords[1]),
            "lon_deg": float(coords[0]),
        })

    # Deduplicate by site_id (keep first occurrence)
    seen: set[str] = set()
    unique_pops: list[dict[str, Any]] = []
    for p in pops:
        if p["site_id"] not in seen:
            seen.add(p["site_id"])
            unique_pops.append(p)

    unique_pops.sort(key=lambda p: p["site_id"])
    return unique_pops


# -- GS↔PoP Edges -------------------------------------------------------------


def preprocess_gs_pop_edges(
    stations: list[dict[str, Any]],
    pops: list[dict[str, Any]],
    c_fiber_km_s: float = C_FIBER_KM_S,
    detour_factor: float = 1.1,
) -> list[dict[str, Any]]:
    """Derive GS→PoP connectivity: each GS connects to its nearest PoP.

    Matches StarPerf's approach (nearest-1, not many-to-many).
    Backhaul delay = haversine distance × detour_factor / c_fiber.
    """
    edges: list[dict[str, Any]] = []
    for gs in stations:
        if not gs["enabled"]:
            continue

        best_dist = float("inf")
        best_pop = None
        for pop in pops:
            dist = haversine_km(
                gs["lat_deg"], gs["lon_deg"],
                pop["lat_deg"], pop["lon_deg"],
            )
            if dist < best_dist:
                best_dist = dist
                best_pop = pop

        if best_pop is not None:
            edges.append({
                "gs_id": gs["gs_id"],
                "pop_code": best_pop["code"],
                "distance_km": round(best_dist, 1),
                "backhaul_delay": round(
                    best_dist * detour_factor / c_fiber_km_s * 1000, 3
                ),
                "capacity_gbps": 100.0,
            })

    # Ensure every PoP has at least one GS: for orphan PoPs,
    # connect to the nearest enabled GS
    connected_pops = {e["pop_code"] for e in edges}
    enabled_gs = [gs for gs in stations if gs["enabled"]]
    for pop in pops:
        if pop["code"] in connected_pops:
            continue
        best_dist = float("inf")
        best_gs = None
        for gs in enabled_gs:
            dist = haversine_km(
                pop["lat_deg"], pop["lon_deg"],
                gs["lat_deg"], gs["lon_deg"],
            )
            if dist < best_dist:
                best_dist = dist
                best_gs = gs
        if best_gs is not None:
            edges.append({
                "gs_id": best_gs["gs_id"],
                "pop_code": pop["code"],
                "distance_km": round(best_dist, 1),
                "backhaul_delay": round(
                    best_dist * detour_factor / c_fiber_km_s * 1000, 3
                ),
                "capacity_gbps": 100.0,
            })

    edges.sort(key=lambda e: (e["gs_id"], e["pop_code"]))
    return edges


# -- Fiber graph ---------------------------------------------------------------


def preprocess_fiber_graph() -> dict[str, Any]:
    """Build a combined terrestrial + submarine fiber graph.

    Sources:
    - ITU ground_fibre.json: terrestrial fiber (operational only)
    - submarine_cable/cable-geo.json + landing-point-geo.json: submarine cables

    Nodes are snapped to a 0.01° grid (~1 km) to merge nearby endpoints.
    Edge weight is distance_km.
    """
    # Grid-snap precision: 0.01° ≈ 1 km
    def snap(lat: float, lon: float) -> tuple[float, float]:
        return round(lat, 2), round(lon, 2)

    nodes: dict[tuple[float, float], int] = {}
    node_list: list[dict[str, float]] = []
    edges: list[dict[str, Any]] = []

    def get_node_id(lat: float, lon: float) -> int:
        key = snap(lat, lon)
        if key not in nodes:
            nid = len(node_list)
            nodes[key] = nid
            node_list.append({"node_id": nid, "lat_deg": key[0], "lon_deg": key[1]})
        return nodes[key]

    # --- Terrestrial fiber (ITU) ---
    terr_path = RAW_DIR / "infra/itu_data/ground_fibre.json"
    with terr_path.open() as f:
        terr = json.load(f)

    terr_count = 0
    for feat in terr["features"]:
        props = feat["properties"]
        status = props.get("status", "")
        if status not in ("Operational", "Fibre Operational"):
            continue

        distance_km = props.get("distance")
        if not distance_km or distance_km <= 0:
            continue

        coords = feat["geometry"]["coordinates"]
        all_points: list[list[float]] = []
        for line in coords:
            all_points.extend(line)

        if len(all_points) < 2:
            continue

        # GeoJSON: [lon, lat]
        start = all_points[0]
        end = all_points[-1]
        node_a = get_node_id(start[1], start[0])
        node_b = get_node_id(end[1], end[0])

        if node_a == node_b:
            continue

        edges.append({
            "node_a": node_a,
            "node_b": node_b,
            "distance_km": round(float(distance_km), 1),
        })
        terr_count += 1

    # --- Submarine cables ---
    cable_path = RAW_DIR / "infra/submarine_cable/cable-geo.json"
    with cable_path.open() as f:
        cables = json.load(f)

    sub_count = 0
    for feat in cables["features"]:
        coords = feat["geometry"]["coordinates"]
        # Flatten MultiLineString
        all_points = [pt for line in coords for pt in line]
        if len(all_points) < 2:
            continue

        # GeoJSON: [lon, lat] — compute total cable length from coordinates
        total_km = 0.0
        for i in range(len(all_points) - 1):
            total_km += haversine_km(
                all_points[i][1], all_points[i][0],
                all_points[i + 1][1], all_points[i + 1][0],
            )

        if total_km <= 0:
            continue

        # Connect first and last point
        start = all_points[0]
        end = all_points[-1]
        node_a = get_node_id(start[1], start[0])
        node_b = get_node_id(end[1], end[0])

        if node_a == node_b:
            continue

        edges.append({
            "node_a": node_a,
            "node_b": node_b,
            "distance_km": round(total_km, 1),
        })
        sub_count += 1

    # --- Stitch nearby nodes ---
    # Submarine cable landing points may not snap to the same grid cell
    # as nearby terrestrial fiber endpoints. Connect nodes within 50 km
    # to bridge the gap.
    import numpy as np

    coords_arr = np.array(
        [[n["lat_deg"], n["lon_deg"]] for n in node_list], dtype=np.float64
    )
    existing_edges: set[tuple[int, int]] = set()
    for e in edges:
        a, b = e["node_a"], e["node_b"]
        existing_edges.add((min(a, b), max(a, b)))

    stitch_count = 0
    n_nodes = len(node_list)
    # Only check submarine cable nodes (those added after terr_count phase)
    # by checking nodes that have few connections — more efficient than all-pairs
    adj_count = [0] * n_nodes
    for e in edges:
        adj_count[e["node_a"]] += 1
        adj_count[e["node_b"]] += 1

    # Find leaf nodes (degree <= 2) and connect to nearest other node
    for i in range(n_nodes):
        if adj_count[i] > 2:
            continue
        lat_i, lon_i = coords_arr[i]
        dlat = coords_arr[:, 0] - lat_i
        dlon = coords_arr[:, 1] - lon_i
        dist_sq = dlat * dlat + dlon * dlon
        dist_sq[i] = float("inf")  # skip self

        # Find closest node within ~0.5° (~50 km)
        candidates = np.where(dist_sq < 0.25)[0]  # 0.5² = 0.25
        if len(candidates) == 0:
            continue
        j = int(candidates[np.argmin(dist_sq[candidates])])
        pair = (min(i, j), max(i, j))
        if pair in existing_edges:
            continue

        dist_km = haversine_km(lat_i, lon_i, coords_arr[j, 0], coords_arr[j, 1])
        if dist_km > 50:
            continue

        edges.append({
            "node_a": i,
            "node_b": j,
            "distance_km": round(dist_km, 1),
        })
        existing_edges.add(pair)
        stitch_count += 1

    print(
        f"  fiber graph: {terr_count} terrestrial + {sub_count} submarine"
        f" + {stitch_count} stitch edges"
    )

    return {
        "nodes": node_list,
        "edges": edges,
        "num_nodes": len(node_list),
        "num_edges": len(edges),
    }


# -- Terminals -----------------------------------------------------------------


def preprocess_terminals() -> list[dict[str, Any]]:
    """Raw starlink_probes.json → clean terminal registry."""
    path = RAW_DIR / "probe_trace/starlink_probes.json"
    with path.open() as f:
        raw: list[dict[str, Any]] = json.load(f)

    terminals: list[dict[str, Any]] = []
    for entry in raw:
        coords = entry["geometry"]["coordinates"]
        terminals.append({
            "terminal_id": int(entry["id"]),
            "lat_deg": float(coords[1]),
            "lon_deg": float(coords[0]),
            "country": str(entry["country_code"]),
        })

    terminals.sort(key=lambda t: t["terminal_id"])
    return terminals


# -- Main ----------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  wrote {path} ({path.stat().st_size:,} bytes)")


def main() -> None:
    print("Vantage data preprocessing")
    print(f"  raw:  {RAW_DIR}")
    print(f"  out:  {OUT_DIR}")
    print()

    # Infrastructure
    stations = preprocess_ground_stations()
    pops = preprocess_pops()
    edges = preprocess_gs_pop_edges(stations, pops)

    _write_json(OUT_DIR / "ground_stations.json", stations)
    _write_json(OUT_DIR / "pops.json", pops)
    _write_json(OUT_DIR / "gs_pop_edges.json", edges)

    # Fiber graph
    fiber = preprocess_fiber_graph()
    _write_json(OUT_DIR / "fiber_graph.json", fiber)

    # Terminal registry
    terminals = preprocess_terminals()
    _write_json(OUT_DIR / "terminals.json", terminals)

    # Manifest
    source_files: dict[str, Path] = {
        "starlink_gs.json": RAW_DIR / "infra/gs_pop_peering/ground_staion/starlink_gs.json",
        "starlink_pop.geojson": RAW_DIR / "infra/gs_pop_peering/pop/starlink_pop.geojson",
        "starlink_probes.json": RAW_DIR / "probe_trace/starlink_probes.json",
        "ground_fibre.json": RAW_DIR / "infra/itu_data/ground_fibre.json",
        "cable-geo.json": RAW_DIR / "infra/submarine_cable/cable-geo.json",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "vantage.config.preprocess",
        "source_hashes": {
            name: _sha256(path) for name, path in source_files.items()
        },
        "counts": {
            "ground_stations": len(stations),
            "pops": len(pops),
            "gs_pop_edges": len(edges),
            "terminals": len(terminals),
            "fiber_nodes": fiber["num_nodes"],
            "fiber_edges": fiber["num_edges"],
        },
    }
    _write_json(OUT_DIR / "manifest.json", manifest)

    print()
    print(
        f"Done. {len(stations)} GS, {len(pops)} PoPs, {len(edges)} edges, "
        f"{len(terminals)} terminals, {fiber['num_nodes']} fiber nodes, "
        f"{fiber['num_edges']} fiber edges."
    )


if __name__ == "__main__":
    main()
