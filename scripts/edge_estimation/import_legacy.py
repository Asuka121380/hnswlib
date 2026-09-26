from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.edge_estimation.contracts import atomic_output_dir, file_entry, sha256_file
else:
    from .contracts import atomic_output_dir, file_entry, sha256_file


def legacy_capabilities(source_format: str) -> Mapping[str, object]:
    """Return the deliberately restricted capability contract for old traces."""
    if source_format == "phase1_p3_events":
        return {"quality": True, "ordered_timing": False,
                "sampling_modulus": 16, "source_boundaries": False}
    if source_format == "PQREPL01":
        return {"quality": False, "ordered_timing": False,
                "legacy_cost_replay_only": True}
    if source_format == "baseline_dco":
        return {"quality": True, "ordered_timing": False,
                "requires_slot_resolution": True,
                "requires_completeness_audit": True}
    raise ValueError(f"unsupported legacy source format: {source_format}")


def import_jsonl(source: Path, source_format: str, output: Path) -> None:
    capabilities = legacy_capabilities(source_format)
    rows = []
    with source.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"legacy row {number} is not an object")
            rows.append(value)
    with atomic_output_dir(output) as partial:
        projection = partial / "events.quality.jsonl"
        with projection.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        manifest = partial / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "capture_mode": "legacy_import",
            "source_format": source_format,
            "source": {"path": str(source.resolve()), "size": source.stat().st_size,
                       "sha256": sha256_file(source)},
            "row_count": len(rows),
            "capabilities": capabilities,
            "unknown_fields_are_null": True,
            "fabricated_fields": [],
            "files": {"events.quality.jsonl": file_entry(projection)},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "complete.json").write_text(json.dumps({
            "schema_version": 1, "stage": "legacy_import",
            "manifest": file_entry(manifest),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a legacy quality-only trace")
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-format", required=True,
                        choices=["phase1_p3_events", "PQREPL01", "baseline_dco"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    import_jsonl(Path(args.source), args.source_format, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
