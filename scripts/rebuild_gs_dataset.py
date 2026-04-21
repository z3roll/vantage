"""Rebuild ground_stations.json + gs_pop_edges.json from Starlink KML.

Inputs:
    data/infra/gs_pop_peering/ground_staion/starlink_gateways_pops.kml
    src/vantage/config/pops.json   (existing 49 PoPs, will drop bom)

Output (overwrites):
    src/vantage/config/pops.json           (48 PoPs — bom dropped)
    src/vantage/config/ground_stations.json
    src/vantage/config/gs_pop_edges.json
    src/vantage/config/manifest.json

Rules (per user, 2026-04-22):
    * Drop PoP ``bom`` (no Starlink service in India).
    * **Every Live KML GS is used** (the 150 km radius rule dropped 63%
      of the 274 Live gateways — unrealistic for real Starlink backhaul
      geography).
    * **Each GS attaches to exactly ONE PoP — its geographic nearest.**
      Backhaul delay is computed from that distance × fiber detour;
      far-away GSs (e.g., Pacific-island sites that end up on syd/lax)
      naturally carry a high backhaul penalty that the TE plane can
      reason about. No radius cap — the physics does the filtering.
    * No special-case overrides. Every PoP in the 48-PoP set receives
      ≥1 Live GS from the nearest-rule sweep (verified empirically —
      see the preview on 2026-04-22).

Per-GS metadata (num_antennas, capacity, etc.) is fixed at the project
default (8 antennas × 20 Gbps = 160 Gbps per GS) since the KML doesn't
expose hardware specs uniformly.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KML_PATH = ROOT / "data/infra/gs_pop_peering/ground_staion/starlink_gateways_pops.kml"
CONFIG_DIR = ROOT / "src/vantage/config"

NS = {"kml": "http://www.opengis.net/kml/2.2"}
LIVE_COLORS = {"558B2F", "388E3C", "097138", "0F9D58"}
PLANNED_COLORS = {"673AB7", "9C27B0", "BDBDBD"}  # Purple/Grey

DROP_POPS = {"bom"}

# Default GS hardware spec — matches the project's "1 PoP = 1 GS = 8 Ka
# antennas × 20 Gbps" convention. Real per-GS variation isn't exposed
# by the KML, so we pick the canonical value used elsewhere.
GS_DEFAULTS = dict(
    num_antennas=8,
    ka_antennas=8,
    e_antennas=0,
    min_elevation_deg=25.0,
    enabled=True,
    uplink_ghz=2.1,
    downlink_ghz=1.3,
    max_capacity=160.0,
    temporary=False,
)

C_FIBER_KM_PER_MS = 200.0  # speed of light in fiber, km/ms
FIBER_DETOUR = 1.4         # actual route length / great-circle distance
PER_GS_CAPACITY_GBPS = 100.0   # backhaul circuit cap (legacy schema field)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def slugify(name: str) -> str:
    """KML name → filesystem-safe gs_id (lowercase, non-alnum → '_')."""
    cleaned = []
    for ch in name.lower():
        cleaned.append(ch if ch.isalnum() else "_")
    out = "".join(cleaned).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out


def parse_kml() -> tuple[
    list[tuple[str, float, float, str, str]],     # (name, lat, lon, color, country)
    set[str],                                       # PoP site_ids found as KML markers
]:
    """Return (placemarks, pop_site_ids_in_kml).

    placemarks excludes Placemarks whose name's site_id matches a known
    PoP — those are PoP markers, not gateways.
    """
    tree = ET.parse(KML_PATH)
    root = tree.getroot()

    style_map: dict[str, str] = {}
    for sm in root.iter(f"{{{NS['kml']}}}StyleMap"):
        sid = sm.get("id", "")
        norm = sm.find(".//kml:Pair[kml:key='normal']/kml:styleUrl", NS)
        if norm is not None and norm.text:
            style_map[sid] = norm.text.lstrip("#")
    style_color: dict[str, str] = {}
    for st in root.iter(f"{{{NS['kml']}}}Style"):
        sid = st.get("id", "")
        c = st.find(".//kml:IconStyle/kml:color", NS)
        if c is not None and c.text:
            s = c.text
            style_color[sid] = (s[6:8] + s[4:6] + s[2:4]).upper()

    pops_existing = json.loads((CONFIG_DIR / "pops.json").read_text())
    pop_site_ids = {p["site_id"].upper() for p in pops_existing}

    placemarks: list[tuple[str, float, float, str, str]] = []
    pop_markers_found: set[str] = set()
    for pm in root.iter(f"{{{NS['kml']}}}Placemark"):
        if pm.find(".//kml:Point", NS) is None:
            continue
        su = pm.find("kml:styleUrl", NS)
        if su is None or not su.text:
            continue
        sid = style_map.get(su.text.lstrip("#"), su.text.lstrip("#"))
        color = style_color.get(sid, "?")

        name_el = pm.find("kml:name", NS)
        name = (name_el.text or "").strip() if name_el is not None else ""
        coord = pm.find(".//kml:Point/kml:coordinates", NS)
        if coord is None or not coord.text:
            continue
        parts = coord.text.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            lon, lat = float(parts[0]), float(parts[1])
        except ValueError:
            continue

        head = name.split(" - ")
        if len(head) >= 2 and head[1].split()[0].upper() in pop_site_ids:
            pop_markers_found.add(head[1].split()[0].upper())
            continue

        # Country = last token after last comma in the name (heuristic).
        country = name.rsplit(",", 1)[-1].strip().split("|")[0].strip()
        if len(country) > 30:
            country = ""

        placemarks.append((name, lat, lon, color, country))

    return placemarks, pop_markers_found


def main() -> None:
    print(f"Reading KML: {KML_PATH}")
    placemarks, pop_kml = parse_kml()
    n_live = sum(1 for _, _, _, c, _ in placemarks if c in LIVE_COLORS)
    n_planned = sum(1 for _, _, _, c, _ in placemarks if c in PLANNED_COLORS)
    print(f"  KML markers: {len(placemarks)} total "
          f"(Live: {n_live}, Planned: {n_planned})")
    print(f"  PoP markers matched in KML: {len(pop_kml)}")

    pops_in = json.loads((CONFIG_DIR / "pops.json").read_text())
    pops_out = [p for p in pops_in if p["code"] not in DROP_POPS]
    print(f"\nPoPs: {len(pops_in)} → {len(pops_out)} (dropped: {sorted(DROP_POPS)})")

    # ── Assign each Live GS to its nearest PoP (1:1 edge) ────────────
    # Each Live placemark gets exactly one parent PoP — whichever has
    # the smallest haversine distance. No radius cap: remote Pacific
    # islands still count, they just carry the backhaul penalty their
    # distance implies.
    live_placemarks = [
        (n, glat, glon, c, country)
        for n, glat, glon, c, country in placemarks
        if c in LIVE_COLORS
    ]
    gs_for_pop: dict[str, list[tuple[str, float, float, str, str, float]]] = {
        p["code"]: [] for p in pops_out
    }
    for n, glat, glon, c, country in live_placemarks:
        best_km = float("inf")
        best_code = None
        for pop in pops_out:
            d = haversine_km(glat, glon, pop["lat_deg"], pop["lon_deg"])
            if d < best_km:
                best_km = d
                best_code = pop["code"]
        if best_code is None:
            continue
        gs_for_pop[best_code].append((n, glat, glon, c, country, best_km))

    for code, picks in gs_for_pop.items():
        picks.sort(key=lambda x: x[5])
        if not picks:
            # Should not happen with 274 Live GSs across 48 PoPs, but
            # surface loudly if it does so we don't silently ship a
            # ground-less PoP.
            print(f"  WARN: {code} received NO Live GS under nearest-rule")

    # ── Build ground_stations.json (deduped by gs_id) ────────────────
    seen: dict[str, dict] = {}
    edges: list[dict] = []
    name_to_gs_id: dict[str, str] = {}

    for code, picks in gs_for_pop.items():
        for name, glat, glon, color, country, dist in picks:
            gs_id = name_to_gs_id.get(name)
            if gs_id is None:
                base = slugify(name)
                gs_id = base
                suffix = 2
                while gs_id in seen and (
                    abs(seen[gs_id]["lat_deg"] - glat) > 1e-4
                    or abs(seen[gs_id]["lon_deg"] - glon) > 1e-4
                ):
                    gs_id = f"{base}_{suffix}"
                    suffix += 1
                name_to_gs_id[name] = gs_id
                if gs_id not in seen:
                    seen[gs_id] = dict(
                        gs_id=gs_id,
                        lat_deg=glat,
                        lon_deg=glon,
                        country=country,
                        town=name,
                        **GS_DEFAULTS,
                    )
            backhaul_ms = (dist * FIBER_DETOUR / C_FIBER_KM_PER_MS)
            edges.append(dict(
                gs_id=gs_id,
                pop_code=code,
                distance_km=round(dist, 2),
                backhaul_delay=round(backhaul_ms, 4),
                capacity_gbps=PER_GS_CAPACITY_GBPS,
            ))

    # Dedupe edges: the KML occasionally has two placemarks at identical
    # coordinates (e.g., Willemstad, CW listed twice). They collapse to
    # one GS via the slugify step but still produce two identical edges;
    # dedupe so the data plane doesn't double-enumerate the same path.
    seen_edge: set[tuple[str, str]] = set()
    deduped_edges: list[dict] = []
    for e in edges:
        key = (e["gs_id"], e["pop_code"])
        if key in seen_edge:
            continue
        seen_edge.add(key)
        deduped_edges.append(e)
    edges = deduped_edges

    gs_records = sorted(seen.values(), key=lambda g: g["gs_id"])
    edges.sort(key=lambda e: (e["pop_code"], e["distance_km"]))

    # ── Write outputs ────────────────────────────────────────────────
    (CONFIG_DIR / "pops.json").write_text(json.dumps(pops_out, indent=2))
    (CONFIG_DIR / "ground_stations.json").write_text(json.dumps(gs_records, indent=2))
    (CONFIG_DIR / "gs_pop_edges.json").write_text(json.dumps(edges, indent=2))

    manifest = dict(
        schema_version=3,
        generated_at=datetime.now().strftime("%Y-%m-%d"),
        generator=f"{Path(__file__).name} (KML Live + special-case rules)",
        counts=dict(
            ground_stations=len(gs_records),
            pops=len(pops_out),
            gs_pop_edges=len(edges),
        ),
    )
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote:")
    print(f"  pops.json:           {len(pops_out):>4} records")
    print(f"  ground_stations.json:{len(gs_records):>4} records")
    print(f"  gs_pop_edges.json:   {len(edges):>4} records")
    print(f"  manifest.json:       schema_version=3")

    # ── Summary ──────────────────────────────────────────────────────
    print("\nPer-PoP GS count:")
    from collections import Counter
    dist = Counter(len(picks) for picks in gs_for_pop.values())
    for n in sorted(dist):
        print(f"  {n} GS  →  {dist[n]} PoPs")


if __name__ == "__main__":
    main()
