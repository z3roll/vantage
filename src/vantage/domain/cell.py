"""Geographic cell domain types.

A ``Cell`` is a static, discrete patch of Earth surface used as the
aggregation unit for traffic-engineering decisions. We use `Uber's H3
<https://h3geo.org>`_ hexagonal grid at resolution 5 (~252 km² per cell)
to match the cell definition used by Starlink (see Mike Puchol,
"Modeling Starlink capacity").

Conventions:
    * ``CellId`` is the 64-bit int form of an H3 index. It is hashable
      and cheap to store. Use :func:`latlng_to_cell_id` / :func:`cell_id_to_str`
      to convert at boundaries.
    * ``Cell`` is a frozen dataclass carrying the id and the center
      (lat, lon) of the hex, pre-computed once.
    * ``CellGrid`` is an immutable lookup structure that (a) holds every
      ``Cell`` used by a simulation run, and (b) memoizes the
      endpoint-name → cell_id mapping so downstream code can resolve
      a flow's source cell in O(1).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import h3

__all__ = [
    "CELL_RESOLUTION",
    "Cell",
    "CellGrid",
    "CellId",
    "cell_id_to_str",
    "latlng_to_cell_id",
]


# H3 resolution used throughout vantage. Res 5 → ~252 km² average hex area,
# matching Starlink's operational cell definition per Mike Puchol's analysis.
CELL_RESOLUTION: int = 5

# A CellId is the 64-bit int form of an H3 index.
CellId = int


def latlng_to_cell_id(lat_deg: float, lon_deg: float, resolution: int = CELL_RESOLUTION) -> CellId:
    """Resolve a geographic point to its containing H3 cell id.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        resolution: H3 resolution (default :data:`CELL_RESOLUTION`).

    Returns:
        The 64-bit int form of the H3 index. H3 v4's native ``latlng_to_cell``
        returns a hex string; we normalize to int for hashing efficiency.
    """
    token = h3.latlng_to_cell(lat_deg, lon_deg, resolution)
    return h3.str_to_int(token)


def cell_id_to_str(cell_id: CellId) -> str:
    """Render a :data:`CellId` as its canonical H3 hex string.

    Useful for logging, serialization, and H3 library calls that expect
    the string form.
    """
    return h3.int_to_str(cell_id)


@dataclass(frozen=True, slots=True)
class Cell:
    """A geographic cell with precomputed center coordinates.

    Coordinates are in WGS-84 degrees. ``cell_id`` is the int form of
    the H3 index (see :data:`CellId`).
    """

    cell_id: CellId
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True, slots=True)
class CellGrid:
    """Immutable lookup structure for all cells used by a simulation run.

    Holds two maps:
        * ``cells``: :data:`CellId` → :class:`Cell`. The authoritative
          per-cell metadata.
        * ``endpoint_to_cell``: endpoint-name → :data:`CellId`. Populated
          once at init so downstream code can resolve a flow's source cell
          without re-running H3.

    Use :meth:`from_endpoints` as the constructor. Direct construction
    is for deserialization only.
    """

    cells: Mapping[CellId, Cell]
    endpoint_to_cell: Mapping[str, CellId]

    @classmethod
    def from_endpoints(
        cls,
        endpoints: Iterable[tuple[str, float, float]],
        resolution: int = CELL_RESOLUTION,
    ) -> CellGrid:
        """Build a grid from ``(name, lat, lon)`` triples.

        Any endpoints that fall into the same hex share a :class:`Cell`.
        The cell's stored ``(lat_deg, lon_deg)`` is the hex center — not
        any one endpoint — so it's stable across runs.
        """
        cells: dict[CellId, Cell] = {}
        endpoint_to_cell: dict[str, CellId] = {}
        for name, lat, lon in endpoints:
            cid = latlng_to_cell_id(lat, lon, resolution)
            if cid not in cells:
                center_lat, center_lon = h3.cell_to_latlng(cell_id_to_str(cid))
                cells[cid] = Cell(cell_id=cid, lat_deg=center_lat, lon_deg=center_lon)
            endpoint_to_cell[name] = cid
        return cls(
            cells=MappingProxyType(cells),
            endpoint_to_cell=MappingProxyType(endpoint_to_cell),
        )

    def cell_of(self, endpoint_name: str) -> CellId:
        """Return the :data:`CellId` hosting ``endpoint_name``.

        Raises :class:`KeyError` if the endpoint was not registered at
        construction time.
        """
        return self.endpoint_to_cell[endpoint_name]

    def __len__(self) -> int:
        return len(self.cells)

    def __contains__(self, cell_id: object) -> bool:
        return cell_id in self.cells
