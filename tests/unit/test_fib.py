"""Unit tests for ``vantage.domain.fib``."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from vantage.domain.fib import (
    ROUTING_PLANE_REFRESH_S,
    CellToPopTable,
    FIBEntry,
    FIBEntryKind,
    RoutingPlane,
    SatelliteFIB,
)


@pytest.mark.unit
class TestFIBEntryVariants:
    """Both variants store their target correctly and reject cross-access."""

    def test_egress_constructor(self) -> None:
        entry = FIBEntry.egress("gs_seattle", cost_ms=12.5)
        assert entry.kind is FIBEntryKind.EGRESS
        assert entry.is_egress
        assert not entry.is_forward
        assert entry.egress_gs == "gs_seattle"
        assert entry.cost_ms == 12.5

    def test_forward_constructor(self) -> None:
        entry = FIBEntry.forward(next_hop_sat=42, cost_ms=7.25)
        assert entry.kind is FIBEntryKind.FORWARD
        assert entry.is_forward
        assert not entry.is_egress
        assert entry.next_hop_sat == 42
        assert entry.cost_ms == 7.25

    def test_egress_access_on_forward_raises(self) -> None:
        entry = FIBEntry.forward(next_hop_sat=42, cost_ms=1.0)
        with pytest.raises(ValueError, match="not EGRESS"):
            _ = entry.egress_gs

    def test_forward_access_on_egress_raises(self) -> None:
        entry = FIBEntry.egress("gs_a", cost_ms=1.0)
        with pytest.raises(ValueError, match="not FORWARD"):
            _ = entry.next_hop_sat

    def test_frozen(self) -> None:
        entry = FIBEntry.egress("gs_a", cost_ms=1.0)
        with pytest.raises(AttributeError):
            entry.cost_ms = 2.0  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = FIBEntry.forward(1, cost_ms=3.0)
        b = FIBEntry.forward(1, cost_ms=3.0)
        assert a == b
        assert hash(a) == hash(b)

    def test_post_init_rejects_egress_with_int_target(self) -> None:
        with pytest.raises(TypeError, match="EGRESS"):
            FIBEntry(kind=FIBEntryKind.EGRESS, target=42, cost_ms=1.0)

    def test_post_init_rejects_forward_with_str_target(self) -> None:
        with pytest.raises(TypeError, match="FORWARD"):
            FIBEntry(kind=FIBEntryKind.FORWARD, target="gs_a", cost_ms=1.0)


@pytest.mark.unit
class TestSatelliteFIB:
    """Lookup semantics and version metadata."""

    def _fib(self, sat_id: int = 1) -> SatelliteFIB:
        return SatelliteFIB(
            sat_id=sat_id,
            fib=MappingProxyType(
                {
                    "pop_tokyo": FIBEntry.forward(next_hop_sat=2, cost_ms=5.0),
                    "pop_seattle": FIBEntry.egress(gs_id="gs_sea", cost_ms=12.0),
                }
            ),
            version=3,
            built_at=100.0,
        )

    def test_route_returns_entry(self) -> None:
        fib = self._fib()
        entry = fib.route("pop_tokyo")
        assert entry.is_forward
        assert entry.next_hop_sat == 2

    def test_route_missing_raises(self) -> None:
        fib = self._fib()
        with pytest.raises(KeyError):
            fib.route("pop_nowhere")

    def test_frozen(self) -> None:
        fib = self._fib()
        with pytest.raises(AttributeError):
            fib.version = 4  # type: ignore[misc]

    def test_live_dict_frozen_on_construction(self) -> None:
        """Mutating the dict passed to the ctor must not affect the FIB."""
        live: dict[str, FIBEntry] = {
            "pop_tokyo": FIBEntry.forward(next_hop_sat=2, cost_ms=5.0),
        }
        fib = SatelliteFIB(sat_id=1, fib=live, version=1, built_at=0.0)
        live["pop_tokyo"] = FIBEntry.forward(next_hop_sat=999, cost_ms=9999.0)
        # FIB unchanged.
        assert fib.route("pop_tokyo").next_hop_sat == 2


@pytest.mark.unit
class TestCellToPopTable:
    """Lookup and immutability."""

    def test_pop_of(self) -> None:
        table = CellToPopTable(
            mapping=MappingProxyType({
                111: ("pop_tokyo", "pop_osaka"),
                222: ("pop_seattle",),
            }),
            version=1,
            built_at=0.0,
        )
        # pop_of returns the head of the ranked tuple.
        assert table.pop_of(111) == "pop_tokyo"
        assert table.pop_of(222) == "pop_seattle"
        # pops_of returns the full ranked cascade.
        assert table.pops_of(111) == ("pop_tokyo", "pop_osaka")
        assert table.pops_of(222) == ("pop_seattle",)

    def test_per_dest_override(self) -> None:
        table = CellToPopTable(
            mapping=MappingProxyType({111: ("pop_tokyo", "pop_osaka")}),
            version=1,
            built_at=0.0,
            per_dest=MappingProxyType({
                (111, "google"): ("pop_osaka", "pop_tokyo"),
            }),
        )
        # Per-dest override wins for matching dest; default cascade for others.
        assert table.pops_of(111, "google") == ("pop_osaka", "pop_tokyo")
        assert table.pop_of(111, "google") == "pop_osaka"
        assert table.pops_of(111, "netflix") == ("pop_tokyo", "pop_osaka")

    def test_missing_raises(self) -> None:
        table = CellToPopTable(
            mapping=MappingProxyType({}),
            version=1,
            built_at=0.0,
        )
        with pytest.raises(KeyError):
            table.pop_of(999)

    def test_live_dict_frozen_on_construction(self) -> None:
        live: dict[int, tuple[str, ...]] = {111: ("pop_tokyo",)}
        table = CellToPopTable(mapping=live, version=1, built_at=0.0)
        live[111] = ("pop_hijacked",)
        assert table.pop_of(111) == "pop_tokyo"
        assert table.pops_of(111) == ("pop_tokyo",)


@pytest.mark.unit
class TestRoutingPlaneStaleness:
    """The plane is stale iff ``now - built_at >= cadence``."""

    def _plane(self, built_at: float = 0.0) -> RoutingPlane:
        return RoutingPlane(
            cell_to_pop=CellToPopTable(
                mapping=MappingProxyType({}),
                version=1,
                built_at=built_at,
            ),
            sat_fibs=MappingProxyType({}),
            version=1,
            built_at=built_at,
        )

    def test_default_cadence_is_fifteen_seconds(self) -> None:
        assert ROUTING_PLANE_REFRESH_S == 15.0

    def test_fresh_immediately_after_build(self) -> None:
        plane = self._plane(built_at=100.0)
        assert not plane.is_stale(now_s=100.0)
        assert not plane.is_stale(now_s=114.999)

    def test_stale_at_cadence_boundary(self) -> None:
        plane = self._plane(built_at=100.0)
        # Exactly at the cadence the plane is considered stale (>=, not >).
        assert plane.is_stale(now_s=115.0)
        assert plane.is_stale(now_s=1000.0)

    def test_custom_cadence(self) -> None:
        plane = self._plane(built_at=100.0)
        assert not plane.is_stale(now_s=130.0, cadence_s=60.0)
        assert plane.is_stale(now_s=161.0, cadence_s=60.0)

    def test_fib_of_returns_entry(self) -> None:
        fib = SatelliteFIB(
            sat_id=7,
            fib=MappingProxyType({}),
            version=1,
            built_at=0.0,
        )
        plane = RoutingPlane(
            cell_to_pop=CellToPopTable(
                mapping=MappingProxyType({}),
                version=1,
                built_at=0.0,
            ),
            sat_fibs=MappingProxyType({7: fib}),
            version=1,
            built_at=0.0,
        )
        assert plane.fib_of(7) is fib
        with pytest.raises(KeyError):
            plane.fib_of(999)

    def test_frozen(self) -> None:
        plane = self._plane()
        with pytest.raises(AttributeError):
            plane.version = 2  # type: ignore[misc]

    def test_live_sat_fibs_frozen_on_construction(self) -> None:
        fib = SatelliteFIB(
            sat_id=7,
            fib=MappingProxyType({}),
            version=1,
            built_at=0.0,
        )
        live: dict[int, SatelliteFIB] = {7: fib}
        plane = RoutingPlane(
            cell_to_pop=CellToPopTable(
                mapping=MappingProxyType({}),
                version=1,
                built_at=0.0,
            ),
            sat_fibs=live,
            version=1,
            built_at=0.0,
        )
        # Tamper with the live dict afterwards.
        live[999] = fib
        # Plane's view is unaffected.
        with pytest.raises(KeyError):
            plane.fib_of(999)
        assert plane.fib_of(7) is fib
