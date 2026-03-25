"""Satellite cost table precomputation.

Computes the minimum satellite segment cost (ISL + downlink + backhaul)
from each ingress satellite to each PoP. Computed once per epoch/snapshot,
shared across all flows.
"""

from __future__ import annotations

import numpy as np

from vantage.common import access_delay
from vantage.domain import NetworkSnapshot


def precompute_sat_cost(
    snapshot: NetworkSnapshot,
) -> dict[tuple[int, str], float]:
    """Precompute sat_cost[ingress_sat, pop_code] → min satellite RTT (ms).

    For each (ingress_sat, PoP), finds the best (GS, egress_sat) that
    minimizes: ISL_rtt + downlink_rtt + backhaul_rtt.

    Uses vectorized numpy for the ISL delay lookup (the hot path).
    """
    sat = snapshot.satellite
    n_sats = sat.num_sats

    # Result: per-PoP minimum cost from every ingress satellite
    # Initialize to infinity, then take min over all (GS, egress) options per PoP
    pop_codes = [p.code for p in snapshot.infra.pops]
    pop_best = {code: np.full(n_sats, np.inf) for code in pop_codes}

    for pop in snapshot.infra.pops:
        for gs_id, backhaul in snapshot.infra.pop_gs_edges(pop.code):
            gs = snapshot.infra.gs_by_id(gs_id)
            if gs is None:
                continue
            gs_links = sat.gateway_attachments.attachments.get(gs_id)
            if not gs_links:
                continue
            backhaul_rtt = backhaul * 2

            for link in gs_links:
                egress = link.sat_id
                # Downlink delay: ECEF computation (done once per GS-satellite pair)
                downlink_rtt = access_delay(
                    gs.lat_deg, gs.lon_deg,
                    float(sat.positions[egress, 0]),
                    float(sat.positions[egress, 1]),
                    float(sat.positions[egress, 2]),
                ) * 2

                # ISL delay from ALL ingress satellites to this egress — vectorized
                isl_rtt = sat.delay_matrix[:, egress].astype(np.float64) * 2

                # Total cost from every ingress sat through this (GS, egress)
                total = isl_rtt + downlink_rtt + backhaul_rtt

                # Element-wise min
                np.minimum(pop_best[pop.code], total, out=pop_best[pop.code])

    # Convert to dict
    cost: dict[tuple[int, str], float] = {}
    for code, arr in pop_best.items():
        for ingress in range(n_sats):
            v = float(arr[ingress])
            if v < np.inf:
                cost[(ingress, code)] = v

    return cost
