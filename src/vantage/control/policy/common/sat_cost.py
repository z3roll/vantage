"""Satellite routing decision precomputation.

For every ``(ingress_sat, pop)`` pair, figures out **three** things
simultaneously from the current snapshot:

    1. ``cost_ms`` — the minimum round-trip satellite-segment RTT
       (ISL + downlink + backhaul).
    2. ``chosen_egress_sat`` — the egress satellite that achieves that
       minimum (the sat that hands off to a ground station).
    3. ``chosen_gs_id`` — the ground station through which the egress
       satellite reaches that PoP.

All three come from the *same* argmin pass, so they are guaranteed
consistent. :func:`precompute_sat_cost` is a thin adapter that drops the argmin
fields and returns a flat dict — used by :func:`compute_cell_sat_cost`
for per-cell sat cost lookup. :func:`precompute_per_sat_routing`
returns the full argmin (egress sat + GS) used by the FIB builder.

Both functions share the same vectorized inner loop: for each
``(pop, gs, egress)`` triple we compute the cost from *every* ingress
satellite using ``sat.delay_matrix[:, egress]`` as a column slice,
then update the running argmin with an element-wise mask. This keeps
the hot path in numpy and matches the previous implementation's
performance envelope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from vantage.common import access_delay
from vantage.model import NetworkSnapshot

__all__ = [
    "PerSatRouting",
    "precompute_per_sat_routing",
    "precompute_sat_cost",
]


@dataclass(frozen=True, slots=True)
class PerSatRouting:
    """Per-``(ingress_sat, pop)`` routing decisions for one snapshot.

    All four arrays are indexed by ``(pop_code, ingress_sat)``:

        * ``cost_ms[pop_code]`` — ``(n_sats,)`` float64; infinity where
          the PoP is unreachable from an ingress sat.
        * ``egress_sat[pop_code]`` — ``(n_sats,)`` int32; ``-1`` means
          unreachable.
        * ``gs_ids`` — a flat tuple of every GS id used anywhere in the
          chosen paths; ``gs_index[pop_code]`` (``(n_sats,)`` int32,
          ``-1`` for unreachable) indexes into ``gs_ids``.

    The GS indirection keeps the hot loop pure-numpy (ints), with the
    string lookup confined to the consumer side.

    Immutability contract: ``__post_init__`` freezes both the three
    dict containers (into :class:`MappingProxyType`) and the numpy
    arrays themselves (by clearing ``flags.writeable``). Downstream
    consumers get a truly read-only view; to inject a modified
    routing state in tests, construct a fresh :class:`PerSatRouting`
    rather than mutating an existing one.
    """

    cost_ms: Mapping[str, NDArray[np.float64]]
    egress_sat: Mapping[str, NDArray[np.int32]]
    gs_index: Mapping[str, NDArray[np.int32]]
    gs_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, mapping in (
            ("cost_ms", self.cost_ms),
            ("egress_sat", self.egress_sat),
            ("gs_index", self.gs_index),
        ):
            # Make the arrays themselves read-only. We do this on the
            # live references before wrapping the dict — writing after
            # the proxy wrap would require another detour.
            for arr in mapping.values():
                arr.flags.writeable = False
            if not isinstance(mapping, MappingProxyType):
                object.__setattr__(
                    self, field_name, MappingProxyType(dict(mapping))
                )

    def chosen_gs(self, pop_code: str, ingress_sat: int) -> str | None:
        """Return the GS id chosen for ``(ingress_sat, pop_code)``, or None."""
        idx = int(self.gs_index[pop_code][ingress_sat])
        if idx < 0:
            return None
        return self.gs_ids[idx]

    def is_reachable(self, pop_code: str, ingress_sat: int) -> bool:
        """``True`` iff there is any path from ``ingress_sat`` to ``pop_code``."""
        return int(self.egress_sat[pop_code][ingress_sat]) >= 0


def precompute_per_sat_routing(snapshot: NetworkSnapshot) -> PerSatRouting:
    """Compute cost + argmin routing decisions for every (ingress_sat, pop).

    See :class:`PerSatRouting` for the output shape. Core algorithm:
    enumerate every physical ``(pop, gs, egress_sat)`` triple the
    topology permits, compute the total round-trip cost from *all*
    ingress sats to that egress in one numpy column slice, and update
    running ``(cost, egress, gs_index)`` arrays wherever the new
    candidate beats the old minimum.

    **Tie-breaking determinism.** The argmin update uses a strict
    less-than mask (``total < pop_cost``), so on equal-cost candidates
    the *first-encountered* ``(egress_sat, gs_id)`` wins. Iteration
    order is fixed because (a) ``snapshot.infra.pops`` is a tuple,
    (b) ``pop_gs_edges`` is built from an insertion-ordered dict in
    ``GroundInfrastructure.__post_init__``, and (c) each GS's
    ``gateway_attachments.attachments`` entry is a tuple. As long as
    the snapshot is built the same way across epochs, this function
    is deterministic.
    """
    sat = snapshot.satellite
    n_sats = sat.num_sats

    pop_codes = tuple(p.code for p in snapshot.infra.pops)

    # Flat, de-duplicated GS id list. The hot loop only deals with
    # int indices into this tuple; the string → int mapping is resolved
    # once here.
    gs_id_list: list[str] = []
    gs_id_to_idx: dict[str, int] = {}
    for pop in snapshot.infra.pops:
        for gs_id, _backhaul in snapshot.infra.pop_gs_edges(pop.code):
            if gs_id not in gs_id_to_idx:
                gs_id_to_idx[gs_id] = len(gs_id_list)
                gs_id_list.append(gs_id)
    gs_ids: tuple[str, ...] = tuple(gs_id_list)

    cost_ms: dict[str, NDArray[np.float64]] = {
        code: np.full(n_sats, np.inf, dtype=np.float64) for code in pop_codes
    }
    egress_sat: dict[str, NDArray[np.int32]] = {
        code: np.full(n_sats, -1, dtype=np.int32) for code in pop_codes
    }
    gs_index: dict[str, NDArray[np.int32]] = {
        code: np.full(n_sats, -1, dtype=np.int32) for code in pop_codes
    }

    for pop in snapshot.infra.pops:
        pop_cost = cost_ms[pop.code]
        pop_egress = egress_sat[pop.code]
        pop_gs_idx = gs_index[pop.code]

        for gs_id, backhaul in snapshot.infra.pop_gs_edges(pop.code):
            gs = snapshot.infra.gs_by_id(gs_id)
            if gs is None:
                continue
            gs_links = sat.gateway_attachments.attachments.get(gs_id)
            if not gs_links:
                continue
            backhaul_rtt = backhaul * 2
            gs_idx_int = gs_id_to_idx[gs_id]

            for link in gs_links:
                egress = link.sat_id
                # Downlink RTT from this egress to this gs.
                downlink_rtt = access_delay(
                    gs.lat_deg,
                    gs.lon_deg,
                    float(sat.positions[egress, 0]),
                    float(sat.positions[egress, 1]),
                    float(sat.positions[egress, 2]),
                ) * 2

                # ISL RTT from every ingress to this egress — single column
                # slice, vectorized.
                isl_rtt = sat.delay_matrix[:, egress].astype(np.float64) * 2
                total = isl_rtt + downlink_rtt + backhaul_rtt

                # Element-wise argmin update: wherever the new candidate is
                # strictly better, overwrite cost AND record the argmin
                # (egress sat + gs index) so everything stays consistent.
                better = total < pop_cost
                pop_cost[better] = total[better]
                pop_egress[better] = egress
                pop_gs_idx[better] = gs_idx_int

    return PerSatRouting(
        cost_ms=cost_ms,
        egress_sat=egress_sat,
        gs_index=gs_index,
        gs_ids=gs_ids,
    )


def precompute_sat_cost(
    snapshot: NetworkSnapshot,
) -> dict[tuple[int, str], float]:
    """Precompute ``sat_cost[ingress_sat, pop_code] → min satellite RTT (ms)``.

    Backwards-compatible thin adapter over :func:`precompute_per_sat_routing`.
    The argmin info is discarded; only entries with a finite cost are
    returned. The output is identical (up to floating-point reproducibility)
    to the pre-refactor implementation, so existing policies continue to
    work unchanged.
    """
    routing = precompute_per_sat_routing(snapshot)
    cost: dict[tuple[int, str], float] = {}
    n_sats = snapshot.satellite.num_sats
    for pop_code, arr in routing.cost_ms.items():
        for ingress in range(n_sats):
            v = float(arr[ingress])
            if v < np.inf:
                cost[(ingress, pop_code)] = v
    return cost
