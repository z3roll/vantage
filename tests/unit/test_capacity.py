"""Unit tests for ``vantage.domain.capacity_view``.

Covers:

    * :class:`CapacityView` — derived lookup over
      ``ISLEdge.capacity_gbps`` / ``ShellConfig.feeder_capacity_gbps``
      / ``GroundStation.max_capacity`` (the three single sources of
      truth). No values are stored twice.
    * :class:`UsageBook` — mutable per-epoch accounting on top of a
      :class:`CapacityView`, including charge/release, utilization,
      saturation, and residual capacity queries.
    * :class:`vantage.domain.SLAViolation` — severity math.

``SLAViolation`` lives in ``domain/result.py`` alongside the other
flow outcome types; we exercise it here because SLA is part of the
capacity story.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pytest

from vantage.common.constants import DEFAULT_FLOW_CIR_GBPS
from vantage.domain import (
    CapacityView,
    GroundStation,
    ISLEdge,
    ISLGraph,
    SatelliteState,
    ShellConfig,
    SLAViolation,
    UsageBook,
)
from vantage.domain.traffic import FlowKey


# --- Fixture helpers --------------------------------------------------------


def _gs(gs_id: str, num_antennas: int = 9) -> GroundStation:
    """Build a minimal GroundStation with ``max_capacity = num_antennas × 10``."""
    return GroundStation(
        gs_id=gs_id,
        lat_deg=0.0,
        lon_deg=0.0,
        country="US",
        town="",
        num_antennas=num_antennas,
        min_elevation_deg=25.0,
        enabled=True,
        uplink_ghz=2.1,
        downlink_ghz=1.3,
        max_capacity=float(num_antennas) * 10.0,
        temporary=False,
    )


def _shell(feeder_cap: float = 20.0) -> ShellConfig:
    return ShellConfig(
        shell_id=1,
        altitude_km=550.0,
        orbit_cycle_s=5731.0,
        inclination_deg=53.0,
        phase_shift=1,
        num_orbits=2,
        sats_per_orbit=2,
        feeder_capacity_gbps=feeder_cap,
    )


def _sat_state_with_isls(
    num_sats: int,
    isl_edges: list[tuple[int, int, float]],
) -> SatelliteState:
    """Build a minimal SatelliteState whose graph contains exactly the
    given ISLs. ``isl_edges`` is a list of ``(sat_a, sat_b, cap_gbps)``.
    """
    edges = tuple(
        ISLEdge(
            sat_a=a,
            sat_b=b,
            delay=1.0,
            distance_km=1.0,
            link_type="intra_orbit",
            capacity_gbps=cap,
        )
        for a, b, cap in isl_edges
    )
    graph = ISLGraph(shell_id=1, timeslot=0, num_sats=num_sats, edges=edges)
    positions = np.zeros((num_sats, 3), dtype=np.float64)
    positions[:, 2] = 550.0  # altitude
    delay_matrix = np.zeros((num_sats, num_sats), dtype=np.float64)
    predecessor_matrix = np.full((num_sats, num_sats), -1, dtype=np.int32)
    return SatelliteState(
        positions=positions,
        graph=graph,
        delay_matrix=delay_matrix,
        predecessor_matrix=predecessor_matrix,
    )


def _view(
    isl_edges: list[tuple[int, int, float]] | None = None,
    num_sats: int = 4,
    feeder_cap: float = 20.0,
    gs_list: list[GroundStation] | None = None,
) -> CapacityView:
    sat_state = _sat_state_with_isls(
        num_sats=num_sats,
        isl_edges=isl_edges or [(0, 1, 96.0)],
    )
    shell = _shell(feeder_cap=feeder_cap)
    gs_by_id: Mapping[str, GroundStation]
    if gs_list is None:
        gs_by_id = MappingProxyType({"gs_a": _gs("gs_a")})
    else:
        gs_by_id = MappingProxyType({g.gs_id: g for g in gs_list})
    return CapacityView.from_snapshot(sat_state, shell, gs_by_id)


# --- CapacityView -----------------------------------------------------------


@pytest.mark.unit
class TestCapacityViewFromSnapshot:
    """``from_snapshot`` correctly threads existing cap fields into a view."""

    def test_isl_cap_from_edge(self) -> None:
        view = _view(isl_edges=[(0, 1, 96.0), (1, 2, 1000.0)], num_sats=3)
        assert view.isl_cap(0, 1) == 96.0
        assert view.isl_cap(1, 2) == 1000.0

    def test_isl_cap_direction_agnostic(self) -> None:
        view = _view(isl_edges=[(0, 1, 96.0)], num_sats=2)
        assert view.isl_cap(0, 1) == view.isl_cap(1, 0) == 96.0

    def test_missing_isl_raises(self) -> None:
        """A (a, b) pair with no real ISL must raise, not silently return a default."""
        view = _view(isl_edges=[(0, 1, 96.0)], num_sats=3)
        with pytest.raises(KeyError):
            view.isl_cap(0, 2)

    def test_sat_feeder_cap_from_shell(self) -> None:
        view = _view(feeder_cap=20.0)
        assert view.sat_feeder_cap(sat_id=0) == 20.0
        # Signature is future-proofed per sat_id; currently flat.
        assert view.sat_feeder_cap(sat_id=999) == 20.0

    def test_gs_feeder_cap_from_ground_station(self) -> None:
        gs = _gs("gs_x", num_antennas=9)
        view = _view(gs_list=[gs])
        assert view.gs_feeder_cap("gs_x") == 90.0

    def test_gs_feeder_cap_unknown_raises(self) -> None:
        view = _view()
        with pytest.raises(KeyError):
            view.gs_feeder_cap("nonexistent")


@pytest.mark.unit
class TestCapacityViewImmutability:
    """Live-dict tampering must not reach the view."""

    def test_view_is_frozen(self) -> None:
        view = _view()
        with pytest.raises(AttributeError):
            view.sat_feeder_gbps = 0.0  # type: ignore[misc]

    def test_direct_construction_freezes_live_dicts(self) -> None:
        live_isl: dict[tuple[int, int], float] = {(0, 1): 96.0}
        live_gs: dict[str, GroundStation] = {"gs_a": _gs("gs_a")}
        view = CapacityView(
            isl_cap_index=live_isl,
            sat_feeder_gbps=20.0,
            gs_by_id=live_gs,
        )
        live_isl[(0, 1)] = 0.0
        live_gs.clear()
        # View is unaffected.
        assert view.isl_cap(0, 1) == 96.0
        assert view.gs_feeder_cap("gs_a") == 90.0


# --- UsageBook --------------------------------------------------------------


@pytest.mark.unit
class TestUsageBookCharging:
    """Charge/release semantics on top of a :class:`CapacityView`."""

    def _book(self) -> UsageBook:
        return UsageBook(view=_view(isl_edges=[(0, 1, 96.0)], num_sats=2))

    def test_isl_key_normalized(self) -> None:
        assert UsageBook.isl_key(1, 2) == (1, 2)
        assert UsageBook.isl_key(2, 1) == (1, 2)

    def test_charge_isl_accumulates_direction_agnostic(self) -> None:
        book = self._book()
        book.charge_isl(0, 1, 10.0)
        book.charge_isl(1, 0, 5.0)
        assert book.isl_used[(0, 1)] == 15.0

    def test_charge_sat_feeder_accumulates(self) -> None:
        book = self._book()
        book.charge_sat_feeder(0, 3.0)
        book.charge_sat_feeder(0, 4.0)
        assert book.sat_feeder_used[0] == 7.0

    def test_charge_gs_feeder_accumulates(self) -> None:
        book = self._book()
        book.charge_gs_feeder("gs_a", 2.5)
        book.charge_gs_feeder("gs_a", 1.0)
        assert book.gs_feeder_used["gs_a"] == 3.5

    def test_release_reduces(self) -> None:
        book = self._book()
        book.charge_isl(0, 1, 10.0)
        book.release_isl(1, 0, 4.0)
        assert book.isl_used[(0, 1)] == 6.0

    def test_release_clamped_at_zero(self) -> None:
        book = self._book()
        book.charge_sat_feeder(0, 1.0)
        book.release_sat_feeder(0, 100.0)
        assert book.sat_feeder_used[0] == 0.0

    def test_charge_rejects_negative(self) -> None:
        book = self._book()
        with pytest.raises(ValueError, match="non-negative"):
            book.charge_isl(0, 1, -1.0)
        with pytest.raises(ValueError, match="non-negative"):
            book.charge_sat_feeder(0, -1.0)
        with pytest.raises(ValueError, match="non-negative"):
            book.charge_gs_feeder("gs_a", -1.0)

    def test_release_rejects_negative(self) -> None:
        book = self._book()
        with pytest.raises(ValueError, match="non-negative"):
            book.release_isl(0, 1, -1.0)
        with pytest.raises(ValueError, match="non-negative"):
            book.release_sat_feeder(0, -1.0)
        with pytest.raises(ValueError, match="non-negative"):
            book.release_gs_feeder("gs_a", -1.0)


@pytest.mark.unit
class TestUsageBookUtilization:
    """Utilization and saturation predicates query the view for caps."""

    def test_zero_use_zero_util(self) -> None:
        book = UsageBook(view=_view())
        assert book.isl_utilization(0, 1) == 0.0
        assert book.sat_feeder_utilization(0) == 0.0
        assert book.gs_feeder_utilization("gs_a") == 0.0

    def test_utilization_ratio(self) -> None:
        book = UsageBook(view=_view(isl_edges=[(0, 1, 100.0)], num_sats=2))
        book.charge_isl(0, 1, 25.0)
        assert book.isl_utilization(0, 1) == 0.25

    def test_saturated_above_one(self) -> None:
        book = UsageBook(view=_view(feeder_cap=10.0))
        book.charge_sat_feeder(0, 15.0)
        assert book.sat_feeder_utilization(0) == 1.5
        assert book.is_sat_feeder_saturated(0)

    def test_not_saturated_at_exactly_one(self) -> None:
        gs = _gs("gs_a", num_antennas=1)  # 10 Gbps cap
        book = UsageBook(view=_view(gs_list=[gs]))
        book.charge_gs_feeder("gs_a", 10.0)
        assert book.gs_feeder_utilization("gs_a") == 1.0
        assert not book.is_gs_feeder_saturated("gs_a")


@pytest.mark.unit
class TestUsageBookResidualCapacity:
    """``remaining_*`` for fair-share / Progressive Filling solvers."""

    def test_remaining_full_when_empty(self) -> None:
        book = UsageBook(view=_view(isl_edges=[(0, 1, 100.0)], num_sats=2))
        assert book.remaining_isl(0, 1) == 100.0
        assert book.remaining_sat_feeder(0) == 20.0
        assert book.remaining_gs_feeder("gs_a") == 90.0

    def test_remaining_decreases_with_charge(self) -> None:
        book = UsageBook(view=_view(isl_edges=[(0, 1, 100.0)], num_sats=2))
        book.charge_isl(0, 1, 30.0)
        assert book.remaining_isl(0, 1) == 70.0

    def test_remaining_clamped_at_zero_when_oversubscribed(self) -> None:
        book = UsageBook(view=_view(feeder_cap=10.0))
        book.charge_sat_feeder(0, 15.0)
        assert book.remaining_sat_feeder(0) == 0.0

    def test_remaining_isl_direction_agnostic(self) -> None:
        book = UsageBook(view=_view(isl_edges=[(0, 1, 100.0)], num_sats=2))
        book.charge_isl(0, 1, 40.0)
        assert book.remaining_isl(0, 1) == book.remaining_isl(1, 0) == 60.0

    def test_remaining_recovers_after_release(self) -> None:
        """Oversubscribe, release back, confirm residual returns to cap.

        This is the "charge past cap → release → remaining == cap" cycle.
        It guards the invariant that the clamp in ``remaining_*`` is only
        a floor at read time, not a loss of state — the underlying counter
        must be a faithful running tally so a subsequent release can push
        it back below the cap, not just stuck at zero.
        """
        book = UsageBook(view=_view(isl_edges=[(0, 1, 100.0)], num_sats=2))
        book.charge_isl(0, 1, 150.0)
        # While oversubscribed, remaining reports clamped zero.
        assert book.remaining_isl(0, 1) == 0.0
        assert book.is_isl_saturated(0, 1)
        book.release_isl(0, 1, 150.0)
        # After full release, the link is fully available again.
        assert book.remaining_isl(0, 1) == 100.0
        assert not book.is_isl_saturated(0, 1)
        assert book.isl_utilization(0, 1) == 0.0


# --- Default CIR constant ---------------------------------------------------


@pytest.mark.unit
class TestFlowCIRConstant:
    """Lock in the CIR default so policy/analysis code can rely on it."""

    def test_default_cir_is_5_mbps(self) -> None:
        assert DEFAULT_FLOW_CIR_GBPS == 0.005


# --- SLAViolation (lives in domain/result.py) -------------------------------


@pytest.mark.unit
class TestSLAViolation:
    """Shortfall / severity math."""

    def _flow(self) -> FlowKey:
        return FlowKey(src="alice", dst="bob")

    def test_no_shortfall_when_meeting_cir(self) -> None:
        v = SLAViolation(
            flow_key=self._flow(),
            demand_gbps=0.010,
            served_gbps=0.005,
            cir_gbps=0.005,
        )
        assert v.shortfall_gbps == 0.0
        assert v.severity == 0.0

    def test_partial_shortfall(self) -> None:
        v = SLAViolation(
            flow_key=self._flow(),
            demand_gbps=0.010,
            served_gbps=0.003,
            cir_gbps=0.005,
        )
        assert v.shortfall_gbps == pytest.approx(0.002)
        assert v.severity == pytest.approx(0.4)

    def test_total_shortfall_clamped(self) -> None:
        v = SLAViolation(
            flow_key=self._flow(),
            demand_gbps=0.010,
            served_gbps=0.0,
            cir_gbps=0.005,
        )
        assert v.severity == 1.0

    def test_zero_cir_is_zero_severity(self) -> None:
        v = SLAViolation(
            flow_key=self._flow(),
            demand_gbps=0.0,
            served_gbps=0.0,
            cir_gbps=0.0,
        )
        assert v.severity == 0.0
