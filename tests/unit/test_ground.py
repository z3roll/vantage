"""Tests for ground infrastructure loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vantage.world.ground import GroundInfrastructure


@pytest.mark.unit
class TestGroundInfrastructure:
    """Test GroundInfrastructure loading from processed JSON."""

    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        pops = [
            {"site_id": "sttlwax1", "code": "sea", "name": "SEA", "lat_deg": 47.6, "lon_deg": -122.3},
            {"site_id": "chcoilx1", "code": "ord", "name": "ORD", "lat_deg": 41.9, "lon_deg": -87.6},
        ]
        gs = [
            {"gs_id": "gs1", "lat_deg": 47.2, "lon_deg": -119.9, "country": "US",
             "town": "Quincy, WA", "num_antennas": 8, "min_elevation_deg": 25, "enabled": True,
             "uplink_ghz": 2.1, "downlink_ghz": 1.3, "min_capacity": 25000.0, "max_capacity": 32000.0, "temporary": False},
            {"gs_id": "gs2", "lat_deg": 41.5, "lon_deg": -93.8, "country": "US",
             "town": "Des Moines, IA", "num_antennas": 4, "min_elevation_deg": 25, "enabled": True,
             "uplink_ghz": 2.1, "downlink_ghz": 1.3, "min_capacity": 25000.0, "max_capacity": 32000.0, "temporary": False},
        ]
        # Each GS connects to its nearest PoP only
        edges = [
            {"gs_id": "gs1", "pop_code": "sea", "distance_km": 200.0,
             "backhaul_delay": 1.1, "capacity_gbps": 100.0},
            {"gs_id": "gs2", "pop_code": "ord", "distance_km": 450.0,
             "backhaul_delay": 2.475, "capacity_gbps": 100.0},
        ]
        manifest = {"schema_version": 1}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "pops.json").write_text(json.dumps(pops))
        (tmp_path / "ground_stations.json").write_text(json.dumps(gs))
        (tmp_path / "gs_pop_edges.json").write_text(json.dumps(edges))
        return tmp_path

    def test_loads_pops(self, data_dir: Path) -> None:
        infra = GroundInfrastructure(data_dir)
        assert len(infra.pops) == 2

    def test_loads_ground_stations(self, data_dir: Path) -> None:
        infra = GroundInfrastructure(data_dir)
        assert len(infra.ground_stations) == 2

    def test_loads_edges(self, data_dir: Path) -> None:
        infra = GroundInfrastructure(data_dir)
        assert len(infra.gs_pop_edges) == 2

    def test_pop_by_code(self, data_dir: Path) -> None:
        infra = GroundInfrastructure(data_dir)
        sea = infra.pop_by_code("sea")
        assert sea is not None
        assert sea.site_id == "sttlwax1"
        assert infra.pop_by_code("nonexistent") is None

    def test_gs_by_id(self, data_dir: Path) -> None:
        infra = GroundInfrastructure(data_dir)
        gs = infra.gs_by_id("gs1")
        assert gs is not None
        assert gs.town == "Quincy, WA"
        assert infra.gs_by_id("nonexistent") is None

    def test_gs_serving_pop(self, data_dir: Path) -> None:
        infra = GroundInfrastructure(data_dir)
        # PoP "sea" served by gs1 only (nearest-1)
        edges = infra.gs_serving_pop("sea")
        assert len(edges) == 1
        assert edges[0].gs_id == "gs1"

    def test_pops_reachable_from_gs(self, data_dir: Path) -> None:
        infra = GroundInfrastructure(data_dir)
        # gs2 reaches only its nearest PoP (ord)
        edges = infra.pops_reachable_from_gs("gs2")
        assert len(edges) == 1
        assert edges[0].pop_code == "ord"

    def test_frozen_entities(self, data_dir: Path) -> None:
        infra = GroundInfrastructure(data_dir)
        with pytest.raises(AttributeError):
            infra.pops[0].lat_deg = 99.0  # type: ignore[misc]

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="manifest"):
            GroundInfrastructure(tmp_path)

    def test_missing_data_file_raises(self, tmp_path: Path) -> None:
        manifest = {"schema_version": 1}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(FileNotFoundError, match="preprocess"):
            GroundInfrastructure(tmp_path)

    def test_corrupt_json_raises(self, tmp_path: Path) -> None:
        manifest = {"schema_version": 1}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "pops.json").write_text("not json")
        with pytest.raises(ValueError, match="Corrupt JSON"):
            GroundInfrastructure(tmp_path)


@pytest.mark.integration
class TestGroundInfrastructureRealData:
    """Test loading real processed data files."""

    @pytest.fixture
    def infra(self) -> GroundInfrastructure:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
        return GroundInfrastructure(data_dir)

    def test_pop_count(self, infra: GroundInfrastructure) -> None:
        assert len(infra.pops) == 49

    def test_gs_count(self, infra: GroundInfrastructure) -> None:
        assert len(infra.ground_stations) == 165

    def test_edge_count(self, infra: GroundInfrastructure) -> None:
        # 165 GS→nearest PoP + 11 orphan PoP→nearest GS
        assert len(infra.gs_pop_edges) == 176

    def test_most_pops_have_at_least_one_gs(self, infra: GroundInfrastructure) -> None:
        """Most PoPs should have at least one GS within 1500km.

        12 PoPs (Asia/Africa/S.America) lack nearby GS due to Starlink's
        current GS deployment being US/EU-centric.
        """
        # Every PoP must have at least one GS
        for pop in infra.pops:
            assert len(infra.gs_serving_pop(pop.code)) > 0, f"PoP {pop.code} has no GS"

    def test_coordinates_valid(self, infra: GroundInfrastructure) -> None:
        for pop in infra.pops:
            assert -90 <= pop.lat_deg <= 90
            assert -180 <= pop.lon_deg <= 180
        for gs in infra.ground_stations:
            assert -90 <= gs.lat_deg <= 90
            assert -180 <= gs.lon_deg <= 180
