"""Dashboard-facing metric aggregation for simulation epochs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from vantage.common.stats import weighted_mean, weighted_percentile

__all__ = [
    "collect_epoch_summary",
    "compute_breakdown",
    "compute_epoch_compare",
    "compute_pop_compare",
    "compute_theoretical_pop_capacity",
    "country_of",
    "extract_cache_state",
    "round_ms",
]


def country_of(src_name: str) -> str:
    parts = src_name.split("_")
    return parts[1] if len(parts) >= 2 and parts[0] == "city" else "??"


def round_ms(values: Mapping[str, float]) -> dict[str, float]:
    """Round a ``{step: milliseconds}`` mapping for JSON output."""
    return {k: round(float(v), 2) for k, v in values.items()}


def compute_theoretical_pop_capacity(
    pop_gs_list: Mapping[str, list[str]],
    pop_list: list[str],
    *,
    antennas_per_gs: int,
    sat_feeder_cap_gbps: float,
) -> dict[str, float]:
    """Per-PoP display cap = attached GS count times feeder budget."""
    return {
        pop: len(pop_gs_list.get(pop, ())) * antennas_per_gs * sat_feeder_cap_gbps
        for pop in pop_list
    }


def compute_breakdown(result: Any, top_contrib: int = 12, top_sats: int = 12) -> dict:
    pop_country_svc: dict[str, dict[tuple[str, str], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    pop_svc_total: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pop_gs_sat: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for flow in result.flow_outcomes:
        country = country_of(flow.flow_key.src)
        pop_country_svc[flow.pop_code][(country, flow.flow_key.dst)] += flow.demand_gbps
        pop_svc_total[flow.pop_code][flow.flow_key.dst] += flow.demand_gbps
        pop_gs_sat[flow.pop_code][flow.gs_id][int(flow.egress_sat)] += flow.demand_gbps

    out: dict[str, dict] = {}
    for pop in pop_country_svc:
        ranked = sorted(pop_country_svc[pop].items(), key=lambda item: -item[1])
        gs_breakdown = []
        total_sats = 0
        for gs_id, sat_loads in pop_gs_sat[pop].items():
            gs_total = sum(sat_loads.values())
            sats_sorted = sorted(sat_loads.items(), key=lambda item: -item[1])
            gs_breakdown.append({
                "gs_id": gs_id,
                "load": round(gs_total, 2),
                "sats": [
                    {"sat_id": sat_id, "load": round(load, 2)}
                    for sat_id, load in sats_sorted[:top_sats]
                ],
                "n_sats": len(sats_sorted),
            })
            total_sats += len(sats_sorted)
        gs_breakdown.sort(key=lambda item: -item["load"])
        out[pop] = {
            "total": round(sum(load for _, load in ranked), 2),
            "n_distinct": len(ranked),
            "top": [
                {"country": country, "svc": svc, "gbps": round(load, 2)}
                for (country, svc), load in ranked[:top_contrib]
            ],
            "svc_gbps": {
                svc: round(load, 2) for svc, load in pop_svc_total[pop].items()
            },
            "gss": gs_breakdown,
            "n_gss": len(gs_breakdown),
            "n_sats": total_sats,
        }
    return out


def collect_epoch_summary(
    *,
    epoch: int,
    result: Any,
    usage_book: Any,
    pop_capacity: Mapping[str, float],
    svc_names: list[str],
    pop_list: list[str],
    pop_gs_list: Mapping[str, list[str]],
    epoch_s: float,
) -> dict:
    """Aggregate per-flow outcomes into the existing dashboard schema."""
    flows = result.flow_outcomes
    rtts = [flow.total_rtt for flow in flows]
    demands = [flow.demand_gbps for flow in flows]
    rtt_pairs = list(zip(rtts, demands, strict=True))
    svc = defaultdict(
        lambda: {"rtts": [], "sat": [], "gnd": [], "demands": [], "demand": 0.0}
    )
    pop_load = defaultdict(float)
    for flow in flows:
        svc[flow.flow_key.dst]["rtts"].append(flow.total_rtt)
        svc[flow.flow_key.dst]["sat"].append(flow.satellite_rtt)
        svc[flow.flow_key.dst]["gnd"].append(flow.ground_rtt)
        svc[flow.flow_key.dst]["demands"].append(flow.demand_gbps)
        svc[flow.flow_key.dst]["demand"] += flow.demand_gbps
        pop_load[flow.pop_code] += flow.demand_gbps

    gs_over = sum(
        1 for gs_id in usage_book.gs_feeder_used
        if usage_book.gs_feeder_utilization(gs_id) > 1.0
    )
    sf_over = sum(
        1 for sat_id in usage_book.sat_feeder_used
        if usage_book.sat_feeder_utilization(sat_id) > 1.0
    )
    isl_utils = [
        usage_book.isl_utilization(a, b)
        for (a, b) in usage_book.isl_used
    ]
    isl_over = sum(1 for util in isl_utils if util > 1.0)
    max_isl_util = max(isl_utils) if isl_utils else 0.0

    svc_out = {}
    for svc_name in svc_names:
        svc_data = svc[svc_name]
        if svc_data["rtts"]:
            pairs = list(zip(svc_data["rtts"], svc_data["demands"], strict=True))
            svc_out[svc_name] = {
                "mean": round(weighted_mean(svc_data["rtts"], svc_data["demands"]), 2),
                "p95": round(weighted_percentile(pairs, 95), 2),
                "p99": round(weighted_percentile(pairs, 99), 2),
                "sat": round(weighted_mean(svc_data["sat"], svc_data["demands"]), 2),
                "gnd": round(weighted_mean(svc_data["gnd"], svc_data["demands"]), 2),
                "n": len(svc_data["rtts"]),
                "demand_gbps": round(svc_data["demand"], 2),
            }

    all_pops_out = []
    for pop in pop_list:
        load = pop_load.get(pop, 0.0)
        cap = pop_capacity.get(pop, 0)
        all_pops_out.append({
            "code": pop,
            "load": round(load, 1),
            "cap": cap,
            "util": round(load / cap * 100, 1) if cap > 0 else 0,
            "n_gss_total": len(pop_gs_list.get(pop, ())),
        })
    all_pops_out.sort(key=lambda item: -item["load"])

    hh = int(epoch * epoch_s) // 3600
    mm = int(epoch * epoch_s) % 3600 // 60
    ss = int(epoch * epoch_s) % 60
    sat_rtts = [flow.satellite_rtt for flow in flows]
    gnd_rtts = [flow.ground_rtt for flow in flows]
    return {
        "epoch": epoch,
        "time_s": epoch * epoch_s,
        "time_str": f"{hh:02d}:{mm:02d}:{ss:02d}",
        "n_flows": len(flows),
        "demand_gbps": round(sum(demands), 1),
        "mean_rtt": round(weighted_mean(rtts, demands), 2) if rtts else 0,
        "p50_rtt": round(weighted_percentile(rtt_pairs, 50), 2),
        "p95_rtt": round(weighted_percentile(rtt_pairs, 95), 2),
        "p99_rtt": round(weighted_percentile(rtt_pairs, 99), 2),
        "mean_sat": round(weighted_mean(sat_rtts, demands), 2) if flows else 0,
        "mean_gnd": round(weighted_mean(gnd_rtts, demands), 2) if flows else 0,
        "mean_prop": round(
            weighted_mean([flow.propagation_rtt for flow in flows], demands), 2
        ) if flows else 0,
        "mean_queue": round(
            weighted_mean([flow.queuing_rtt for flow in flows], demands), 2
        ) if flows else 0,
        "mean_tx": round(
            weighted_mean([flow.transmission_rtt for flow in flows], demands), 2
        ) if flows else 0,
        "gs_overloaded": gs_over,
        "sf_overloaded": sf_over,
        "isl_overloaded": isl_over,
        "max_isl_util": round(max_isl_util, 4),
        "services": svc_out,
        "pops": all_pops_out,
    }


def compute_pop_compare(result_bl: Any, result_greedy: Any) -> dict:
    bl_rtt_by_fk = {flow.flow_key: flow.total_rtt for flow in result_bl.flow_outcomes}
    bl_demand_by_pop: dict[str, float] = defaultdict(float)
    for flow in result_bl.flow_outcomes:
        bl_demand_by_pop[flow.pop_code] += flow.demand_gbps
    bl_total = sum(bl_demand_by_pop.values()) or 1.0

    greedy_demand_by_pop: dict[str, float] = defaultdict(float)
    greedy_flows_by_pop: dict[str, list] = defaultdict(list)
    for flow in result_greedy.flow_outcomes:
        greedy_demand_by_pop[flow.pop_code] += flow.demand_gbps
        greedy_flows_by_pop[flow.pop_code].append(flow)
    greedy_total = sum(greedy_demand_by_pop.values()) or 1.0

    out: dict[str, dict] = {}
    for pop in set(bl_demand_by_pop) | set(greedy_demand_by_pop):
        sum_w = sum_greedy = sum_bl = 0.0
        for flow in greedy_flows_by_pop.get(pop, ()):
            bl_rtt = bl_rtt_by_fk.get(flow.flow_key)
            if bl_rtt is None:
                continue
            sum_w += flow.demand_gbps
            sum_greedy += flow.total_rtt * flow.demand_gbps
            sum_bl += bl_rtt * flow.demand_gbps
        if sum_w > 0:
            greedy_e2e = sum_greedy / sum_w
            bl_e2e = sum_bl / sum_w
            delta_pct = (greedy_e2e - bl_e2e) / bl_e2e * 100 if bl_e2e else 0.0
        else:
            greedy_e2e = bl_e2e = delta_pct = None
        out[pop] = {
            "bl_pct": round(bl_demand_by_pop.get(pop, 0.0) / bl_total * 100, 2),
            "greedy_pct": round(greedy_demand_by_pop.get(pop, 0.0) / greedy_total * 100, 2),
            "bl_e2e": round(bl_e2e, 2) if bl_e2e is not None else None,
            "greedy_e2e": round(greedy_e2e, 2) if greedy_e2e is not None else None,
            "delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
        }
    return out


def compute_epoch_compare(result_bl: Any, result_greedy: Any, result_lp: Any, result_mip: Any) -> dict:
    """Per-epoch cross-controller metrics using demand-weighted RTTs."""
    bl_by = {flow.flow_key: flow for flow in result_bl.flow_outcomes}
    greedy_by = {flow.flow_key: flow for flow in result_greedy.flow_outcomes}
    lp_by = {flow.flow_key: flow for flow in result_lp.flow_outcomes}
    mip_by = {flow.flow_key: flow for flow in result_mip.flow_outcomes}
    common = set(bl_by) & set(greedy_by) & set(lp_by) & set(mip_by)
    tolerance = 0.5

    total_n = 0
    total_demand = 0.0
    reroutes = {"greedy": [0, 0, 0, 0], "lp": [0, 0, 0, 0], "mip": [0, 0, 0, 0]}
    sums = {
        "bl": [0.0, 0.0],
        "greedy": [0.0, 0.0],
        "lp": [0.0, 0.0],
        "mip": [0.0, 0.0],
    }
    rtt_pairs = {"bl": [], "greedy": [], "lp": [], "mip": []}
    rtt_wsum = {"bl": 0.0, "greedy": 0.0, "lp": 0.0, "mip": 0.0}

    series_by = {"bl": bl_by, "greedy": greedy_by, "lp": lp_by, "mip": mip_by}
    for flow_key in common:
        baseline = bl_by[flow_key]
        demand = baseline.demand_gbps
        total_n += 1
        total_demand += demand
        for key, by_flow in series_by.items():
            flow = by_flow[flow_key]
            rtt_pairs[key].append((flow.total_rtt, demand))
            rtt_wsum[key] += flow.total_rtt * demand
            sums[key][0] += flow.satellite_rtt * demand
            sums[key][1] += flow.ground_rtt * demand
        for key in ("greedy", "lp", "mip"):
            candidate = series_by[key][flow_key]
            if baseline.pop_code != candidate.pop_code:
                reroutes[key][0] += 1
                delta_total = candidate.total_rtt - baseline.total_rtt
                if delta_total < -tolerance:
                    reroutes[key][1] += 1
                elif delta_total > tolerance:
                    reroutes[key][2] += 1
                else:
                    reroutes[key][3] += 1

    def _weighted_avg(value: float) -> float:
        return value / total_demand if total_demand > 0 else 0.0

    out = {
        "total_flows": total_n,
        "total_demand_gbps": round(total_demand, 2),
        "reroute_flows": reroutes["greedy"][0],
        "reroute_pct": round(reroutes["greedy"][0] / total_n * 100, 2) if total_n else 0,
        "impr_flows": reroutes["greedy"][1],
        "worse_flows": reroutes["greedy"][2],
        "neutral_flows": reroutes["greedy"][3],
        "impr_pct_of_re": round(reroutes["greedy"][1] / reroutes["greedy"][0] * 100, 1) if reroutes["greedy"][0] else 0,
        "worse_pct_of_re": round(reroutes["greedy"][2] / reroutes["greedy"][0] * 100, 1) if reroutes["greedy"][0] else 0,
    }
    for key, prefix in (("bl", "bl"), ("greedy", "greedy"), ("lp", "lp"), ("mip", "mip")):
        out[f"{prefix}_mean_rtt"] = round(_weighted_avg(rtt_wsum[key]), 2)
        out[f"{prefix}_sat_rtt"] = round(_weighted_avg(sums[key][0]), 2)
        out[f"{prefix}_gnd_rtt"] = round(_weighted_avg(sums[key][1]), 2)
        out[f"{prefix}_p95_rtt"] = round(weighted_percentile(rtt_pairs[key], 95), 2)
        out[f"{prefix}_p99_rtt"] = round(weighted_percentile(rtt_pairs[key], 99), 2)
    for key, prefix in (("lp", "lp"), ("mip", "mip")):
        reroute_n, improved, worse, neutral = reroutes[key]
        out[f"{prefix}_reroute_flows"] = reroute_n
        out[f"{prefix}_reroute_pct"] = round(reroute_n / total_n * 100, 2) if total_n else 0
        out[f"{prefix}_impr_flows"] = improved
        out[f"{prefix}_worse_flows"] = worse
        out[f"{prefix}_neutral_flows"] = neutral
        out[f"{prefix}_impr_pct_of_re"] = round(improved / reroute_n * 100, 1) if reroute_n else 0
        out[f"{prefix}_worse_pct_of_re"] = round(worse / reroute_n * 100, 1) if reroute_n else 0
    return out


def extract_cache_state(ground_knowledge: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for (pop, dest), rtt in ground_knowledge.all_entries().items():
        if ":" in dest:
            continue
        out.setdefault(pop, {})[dest] = round(rtt, 2)
    return out
