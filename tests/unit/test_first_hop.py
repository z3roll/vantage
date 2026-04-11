"""Unit tests for ``first_hop_on_path``.

The helper walks backward through a predecessor matrix to find the
first ISL hop on a shortest path. Tests cover the happy path, the
``src == dst`` short-circuit, the unreachable case, and a
malformed-matrix guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from vantage.world.satellite.routing import first_hop_on_path


def _predecessor_from_paths(n: int, paths: dict[tuple[int, int], list[int]]) -> np.ndarray:
    """Build a predecessor matrix from an explicit ``(src, dst) → [src, ..., dst]``
    dictionary. Diagonal entries are ``i`` (self), as produced by
    ``compute_all_pairs``.
    """
    pred = np.full((n, n), -1, dtype=np.int32)
    for i in range(n):
        pred[i, i] = i
    for (src, dst), path in paths.items():
        assert path[0] == src and path[-1] == dst, f"invalid path for ({src},{dst})"
        for i in range(1, len(path)):
            pred[src, path[i]] = path[i - 1]
    return pred


@pytest.mark.unit
class TestFirstHopOnPath:
    def test_src_equals_dst_returns_src(self) -> None:
        pred = _predecessor_from_paths(3, {})
        assert first_hop_on_path(pred, 1, 1) == 1

    def test_one_hop(self) -> None:
        """src → dst is a single ISL; first hop == dst."""
        pred = _predecessor_from_paths(3, {(0, 1): [0, 1]})
        assert first_hop_on_path(pred, 0, 1) == 1

    def test_multi_hop_path(self) -> None:
        """Path 0 → 1 → 2 → 3: first hop from 0 to 3 is 1."""
        pred = _predecessor_from_paths(4, {(0, 3): [0, 1, 2, 3]})
        assert first_hop_on_path(pred, 0, 3) == 1

    def test_multi_hop_intermediate(self) -> None:
        """Intermediate target: first hop from 0 to 2 is still 1 (same prefix)."""
        pred = _predecessor_from_paths(4, {(0, 2): [0, 1, 2], (0, 3): [0, 1, 2, 3]})
        assert first_hop_on_path(pred, 0, 2) == 1

    def test_unreachable_returns_minus_one(self) -> None:
        """When predecessor[src, dst] == -1, dst is unreachable from src."""
        pred = np.full((3, 3), -1, dtype=np.int32)
        for i in range(3):
            pred[i, i] = i
        assert first_hop_on_path(pred, 0, 2) == -1

    def test_cycle_in_predecessor_raises(self) -> None:
        """A malformed predecessor matrix with a self-referential cycle
        that does not terminate at ``src`` must be rejected."""
        pred = np.array(
            [
                [0, -1, 1],  # pred[0, 2] = 1 (next backward)
                [-1, 1, -1],
                [-1, 2, 2],  # pred[0, 1] would need to be set; leave it -1
            ],
            dtype=np.int32,
        )
        # Construct a real cycle: pred[0, 2] = 1, pred[0, 1] = 2 → cycle 2 → 1 → 2
        pred[0, 2] = 1
        pred[0, 1] = 2
        with pytest.raises(ValueError, match="did not terminate"):
            first_hop_on_path(pred, 0, 2)
