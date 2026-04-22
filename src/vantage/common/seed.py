"""Run-level seed derivation for fair stochastic comparison.

A single ``run_seed`` (CLI ``--seed`` or a freshly generated integer)
is split into independent sub-seeds per stochastic subsystem. Each
subsystem's sub-seed is deterministic in ``(run_seed, tag)`` and free
of cross-subsystem coupling, so changing traffic randomness does not
perturb ground-delay sampling or ingress-sat selection.

Sub-seeds are also used as run-level *bases* inside ``realize`` and
``GeographicGroundDelay`` to further split by ``(epoch)`` or
``(pop, dest)`` — see :func:`mix_seed` for the call-order-free key
hash used there.
"""

from __future__ import annotations

import hashlib
import secrets

__all__ = ["derive_subseed", "fresh_run_seed", "mix_seed"]

_MASK64 = (1 << 64) - 1


def fresh_run_seed() -> int:
    """Generate a non-deterministic 63-bit run seed."""
    return secrets.randbits(63)


def derive_subseed(run_seed: int, tag: str) -> int:
    """Split ``run_seed`` into a sub-seed for a named subsystem."""
    payload = f"{int(run_seed) & _MASK64}:{tag}".encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "big",
    ) & _MASK64


def mix_seed(base: int, *parts: object) -> int:
    """Hash ``(base, *parts)`` into a 64-bit seed.

    The result depends only on the unordered identity of the parts
    concatenated in the call, so callers that key by e.g.
    ``(pop_code, dest_name)`` or ``(run_base, epoch)`` get the same
    seed regardless of when they ask for it. Used to seed per-key
    local RNGs in place of sequential consumption from a shared
    stream, which is sensitive to call order.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(base) & _MASK64).encode())
    for p in parts:
        h.update(b"\x1f")
        h.update(str(p).encode())
    return int.from_bytes(h.digest(), "big") & _MASK64
