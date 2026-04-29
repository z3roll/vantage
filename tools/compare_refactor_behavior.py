"""Behavior fingerprint for the package-layout refactor.

The script builds a tiny deterministic forwarding scenario and either
writes a JSON fingerprint or compares the current fingerprint with a
previously written baseline. It intentionally imports through both the
new and old package paths so it can run before and after the refactor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


def _imports() -> dict[str, Any]:
    try:
        from vantage.control.feedback import GroundDelayFeedback
        from vantage.control.knowledge import GroundKnowledge
        from vantage.control.plane import CellToPopTable, PopEgressTable, RoutingPlane, SatPathTable
        from vantage.forward import EgressOption, RoutingPlaneForward, realize
        from vantage.forward.execution.context import RunContext
        from vantage.forward.resources.accounting import CapacityView, UsageBook
        from vantage.model.coverage import Cell, CellGrid
        from vantage.model.satellite.state import AccessLink
        from vantage.traffic.types import Endpoint, FlowKey, TrafficDemand

        import vantage.forward.execution.runner as runner_mod

        return {
            "AccessLink": AccessLink,
            "CapacityView": CapacityView,
            "Cell": Cell,
            "CellGrid": CellGrid,
            "CellToPopTable": CellToPopTable,
            "EgressOption": EgressOption,
            "Endpoint": Endpoint,
            "FlowKey": FlowKey,
            "GroundDelayFeedback": GroundDelayFeedback,
            "GroundKnowledge": GroundKnowledge,
            "PopEgressTable": PopEgressTable,
            "RoutingPlane": RoutingPlane,
            "RoutingPlaneForward": RoutingPlaneForward,
            "RunContext": RunContext,
            "SatPathTable": SatPathTable,
            "TrafficDemand": TrafficDemand,
            "UsageBook": UsageBook,
            "realize": realize,
            "runner_mod": runner_mod,
        }
    except ImportError:
        from vantage.domain import (
            AccessLink,
            CapacityView,
            Cell,
            CellGrid,
            CellToPopTable,
            Endpoint,
            FlowKey,
            PopEgressTable,
            RoutingPlane,
            SatPathTable,
            TrafficDemand,
            UsageBook,
        )
        from vantage.engine.context import RunContext
        from vantage.engine.feedback import GroundDelayFeedback
        from vantage.forward import EgressOption, RoutingPlaneForward, realize
        from vantage.world.ground import GroundKnowledge

        import vantage.forward as runner_mod

        return {
            "AccessLink": AccessLink,
            "CapacityView": CapacityView,
            "Cell": Cell,
            "CellGrid": CellGrid,
            "CellToPopTable": CellToPopTable,
            "EgressOption": EgressOption,
            "Endpoint": Endpoint,
            "FlowKey": FlowKey,
            "GroundDelayFeedback": GroundDelayFeedback,
            "GroundKnowledge": GroundKnowledge,
            "PopEgressTable": PopEgressTable,
            "RoutingPlane": RoutingPlane,
            "RoutingPlaneForward": RoutingPlaneForward,
            "RunContext": RunContext,
            "SatPathTable": SatPathTable,
            "TrafficDemand": TrafficDemand,
            "UsageBook": UsageBook,
            "realize": realize,
            "runner_mod": runner_mod,
        }


class _Truth:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, str]] = []

    def sample(self, pop_code: str, dest: str, epoch: int, flow_id: str) -> float:
        self.calls.append((pop_code, dest, epoch, flow_id))
        return 70.0 + sum(ord(c) for c in flow_id) * 0.1


class _GSStub:
    __slots__ = ("max_capacity",)

    def __init__(self, cap: float) -> None:
        self.max_capacity = cap


class _Sat:
    positions = np.zeros((1, 3), dtype=np.float64)


class _Snapshot:
    satellite = _Sat()


def _build_fingerprint() -> dict[str, Any]:
    ns = _imports()
    AccessLink = ns["AccessLink"]
    CapacityView = ns["CapacityView"]
    Cell = ns["Cell"]
    CellGrid = ns["CellGrid"]
    CellToPopTable = ns["CellToPopTable"]
    EgressOption = ns["EgressOption"]
    Endpoint = ns["Endpoint"]
    FlowKey = ns["FlowKey"]
    GroundDelayFeedback = ns["GroundDelayFeedback"]
    GroundKnowledge = ns["GroundKnowledge"]
    PopEgressTable = ns["PopEgressTable"]
    RoutingPlane = ns["RoutingPlane"]
    RoutingPlaneForward = ns["RoutingPlaneForward"]
    RunContext = ns["RunContext"]
    SatPathTable = ns["SatPathTable"]
    TrafficDemand = ns["TrafficDemand"]
    UsageBook = ns["UsageBook"]
    realize = ns["realize"]
    runner_mod = ns["runner_mod"]

    class StubForward(RoutingPlaneForward):
        __slots__ = ()

        def _options_for(
            self,
            ingress: int,
            pop_code: str,
            snapshot: object,
            *,
            opts_by_pop: dict[str, tuple[Any, ...]] | None = None,
        ) -> tuple[Any, ...]:
            del snapshot, opts_by_pop
            return (
                EgressOption(
                    pop_code=pop_code,
                    egress_sat=ingress,
                    gs_id="gs-a",
                    isl_links=(),
                    propagation_rtt=1.0,
                    ground_rtt=0.0,
                ),
            )

    runner_mod.SphericalAccessModel.compute_access = (
        lambda self, lat, lon, alt, sat_positions, min_elev: [
            AccessLink(sat_id=0, elevation_deg=80.0, slant_range_km=100.0, delay=0.5)
        ]
    )
    runner_mod.find_ingress_satellite = (
        lambda src, sat_positions, *, rng=None, _visible=None: _visible[0]
    )

    cell_id = 101
    endpoints = {
        "alice": Endpoint("alice", 0.0, 0.0),
        "bob": Endpoint("bob", 0.0, 0.0),
        "carol": Endpoint("carol", 0.0, 0.0),
    }
    grid = CellGrid(
        cells=MappingProxyType({cell_id: Cell(cell_id=cell_id, lat_deg=0.0, lon_deg=0.0)}),
        endpoint_to_cell=MappingProxyType({name: cell_id for name in endpoints}),
    )
    delay = np.zeros((1, 1), dtype=np.float64)
    pred = np.zeros((1, 1), dtype=np.int32)
    plane = RoutingPlane(
        cell_to_pop=CellToPopTable(
            mapping=MappingProxyType({cell_id: ("pop-a",)}),
            version=7,
            built_at=7.0,
        ),
        sat_paths=SatPathTable(delay_matrix=delay, predecessor_matrix=pred, version=7, built_at=7.0),
        pop_egress=PopEgressTable(candidates=MappingProxyType({}), version=7, built_at=7.0),
        version=7,
        built_at=7.0,
    )
    book = UsageBook(
        view=CapacityView(
            isl_cap_index=MappingProxyType({}),
            sat_feeder_gbps=20.0,
            gs_by_id=MappingProxyType({"gs-a": _GSStub(160.0)}),
        )
    )
    knowledge = GroundKnowledge()
    knowledge.put("pop-a", "dst", 11.1)
    truth = _Truth()
    ctx = RunContext(
        world=object(),
        endpoints=endpoints,
        ground_knowledge=knowledge,
        ground_truth=truth,
    )
    demand = TrafficDemand(
        epoch=5,
        flows=MappingProxyType({
            FlowKey("alice", "dst"): 1.0,
            FlowKey("bob", "dst"): 2.0,
            FlowKey("carol", "dst"): 3.0,
        }),
    )
    result = realize(StubForward(plane, grid, book), _Snapshot(), demand, ctx, ingress_seed_base=123)
    GroundDelayFeedback(knowledge).observe(result)

    outcomes = []
    for f in sorted(result.flow_outcomes, key=lambda x: x.flow_key.src):
        outcomes.append({
            "src": f.flow_key.src,
            "dst": f.flow_key.dst,
            "pop": f.pop_code,
            "gs": f.gs_id,
            "user_sat": f.user_sat,
            "egress_sat": f.egress_sat,
            "satellite_rtt": round(f.satellite_rtt, 9),
            "ground_rtt": round(f.ground_rtt, 9),
            "total_rtt": round(f.total_rtt, 9),
            "demand": round(f.demand_gbps, 9),
            "propagation_rtt": round(f.propagation_rtt, 9),
            "queuing_rtt": round(f.queuing_rtt, 9),
            "transmission_rtt": round(f.transmission_rtt, 9),
            "loss": round(f.loss_probability, 12),
            "bottleneck": round(f.bottleneck_gbps, 9),
        })

    stats = {
        f"{pop}:{dest}": [round(float(x), 9) for x in stat.as_tuple()]
        for (pop, dest), stat in sorted(knowledge.all_stats().items())
    }
    timing = dict(result.forward_timing_ms)
    return {
        "plane": {
            "cell": cell_id,
            "pops": list(plane.cell_to_pop.pops_of(cell_id, "dst")),
            "version": plane.version,
        },
        "outcomes": outcomes,
        "truth_calls": [list(call) for call in truth.calls],
        "knowledge": stats,
        "counts": {
            "n_outcomes": len(result.flow_outcomes),
            "total_demand": round(result.total_demand_gbps, 9),
            "routed_demand": round(result.routed_demand_gbps, 9),
            "unrouted_demand": round(result.unrouted_demand_gbps, 9),
        },
        "timing_keys": sorted(timing),
        "timing_nonnegative": all(float(v) >= 0.0 for v in timing.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    fingerprint = _build_fingerprint()
    if args.write_baseline:
        args.write_baseline.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
        print(f"wrote baseline: {args.write_baseline}")
        return
    if args.baseline:
        expected = json.loads(args.baseline.read_text())
        if fingerprint != expected:
            print(json.dumps({"expected": expected, "actual": fingerprint}, indent=2, sort_keys=True))
            raise SystemExit(1)
        print("fingerprint matches baseline")
        return
    print(json.dumps(fingerprint, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
