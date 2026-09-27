from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.edge_estimation.contracts import (  # type: ignore
        load_strict_json, sha256_file)
else:
    from .contracts import load_strict_json, sha256_file


METHODS = ("pq8", "pq4", "opq", "prq", "jq", "rabitq")


def build_matrix(run_root: Path, *, blocks: int, repeats: int,
                 seed: int, reference: str) -> dict[str, object]:
    if blocks <= 0 or repeats <= 0:
        raise ValueError("blocks and repeats must be positive")
    if reference not in METHODS:
        raise ValueError(f"reference must be one of: {', '.join(METHODS)}")
    methods: list[dict[str, str]] = []
    for method in METHODS:
        artifact = (run_root / f"artifact-{method}").resolve()
        validation = (run_root / f"validation-{method}" / "validation.json").resolve()
        quality = (run_root / f"quality-v2-{method}" / "quality.json").resolve()
        complete = (run_root / f"quality-v2-{method}" / "complete.json").resolve()
        required = (artifact / "manifest.json", artifact / "native.cfg",
                    validation, quality, complete)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{method} timing inputs are missing: {', '.join(missing)}")
        validation_value = load_strict_json(validation)
        if validation_value.get("valid") is not True:
            raise ValueError(f"{method} artifact validation is not passing")
        quality_value = load_strict_json(quality)
        if (quality_value.get("stage") != "quality-sweep" or
                not isinstance(quality_value.get("summaries"), list) or
                not quality_value["summaries"]):
            raise ValueError(f"{method} quality-v2 evidence is invalid")
        complete_value = load_strict_json(complete)
        quality_entry = complete_value.get("outputs", {}).get("quality", {})
        if (complete_value.get("stage") != "quality-sweep" or
                quality_entry.get("path") != quality.name or
                quality_entry.get("size") != quality.stat().st_size or
                quality_entry.get("sha256") != sha256_file(quality)):
            raise ValueError(f"{method} quality-v2 completion identity is invalid")
        methods.append({
            "name": method,
            "artifact": str(artifact),
            "validation": str(validation),
            "quality_report": str(quality),
        })
    return {
        "schema_version": 1,
        "reference": reference,
        "blocks": blocks,
        "repeats": repeats,
        "seed": seed,
        "isa_profile": "portable",
        "methods": methods,
    }


def write_matrix(path: Path, value: object) -> str:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to replace different matrix: {path}")
        return "reused"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return "created"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a six-method formal paired-timing matrix")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--reference", choices=METHODS, default="pq8")
    args = parser.parse_args(argv)
    matrix = build_matrix(args.run_root.resolve(), blocks=args.blocks,
                          repeats=args.repeats, seed=args.seed,
                          reference=args.reference)
    status = write_matrix(args.out, matrix)
    print(json.dumps({"status": status, "out": str(args.out.resolve()),
                      "method_count": len(METHODS),
                      "record_count": args.blocks * args.repeats * len(METHODS)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
