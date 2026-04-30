"""Compare deterministic run.py outputs before and after refactors."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_sim(run_args: str) -> dict[str, Any]:
    cmd = [sys.executable, str(REPO_ROOT / "run.py"), *shlex.split(run_args)]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        raise SystemExit(f"run.py exited with status {proc.returncode}")
    data_path: Path | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("Data: "):
            data_path = Path(line.removeprefix("Data: ").strip())
    if data_path is None:
        print(proc.stdout)
        raise RuntimeError("run.py output did not include a 'Data:' line")
    return json.loads(data_path.read_text())


def _scrub_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            if key == "started_at" or key == "timing":
                continue
            if key.endswith("_ms"):
                continue
            out[key] = _scrub_runtime_fields(child)
        return out
    if isinstance(value, list):
        return [_scrub_runtime_fields(child) for child in value]
    return value


def _first_diffs(a: Any, b: Any, path: str = "", limit: int = 20) -> list[str]:
    if len(path) == 0:
        path = "$"
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, Mapping):
        diffs: list[str] = []
        for key in sorted(set(a) | set(b)):
            child_path = f"{path}.{key}"
            if key not in a:
                diffs.append(f"{child_path}: missing in actual")
            elif key not in b:
                diffs.append(f"{child_path}: missing in expected")
            else:
                diffs.extend(_first_diffs(a[key], b[key], child_path, limit))
            if len(diffs) >= limit:
                return diffs[:limit]
        return diffs
    if isinstance(a, list):
        diffs = []
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
        for i, (left, right) in enumerate(zip(a, b, strict=False)):
            diffs.extend(_first_diffs(left, right, f"{path}[{i}]", limit))
            if len(diffs) >= limit:
                return diffs[:limit]
        return diffs
    if a != b:
        return [f"{path}: actual={a!r} expected={b!r}"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--run-args", required=True)
    args = parser.parse_args()

    fingerprint = _scrub_runtime_fields(_run_sim(args.run_args))

    if args.write_baseline:
        args.write_baseline.write_text(
            json.dumps(fingerprint, indent=2, sort_keys=True),
        )
        print(f"wrote baseline: {args.write_baseline}")
        return

    if args.baseline:
        expected = json.loads(args.baseline.read_text())
        if fingerprint != expected:
            print("run output differs after scrubbing runtime timing fields")
            for diff in _first_diffs(fingerprint, expected):
                print(diff)
            raise SystemExit(1)
        print("run output matches baseline")
        return

    print(json.dumps(fingerprint, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
