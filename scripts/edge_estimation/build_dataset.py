from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.edge_estimation.contracts import read_events, read_labels, read_query_ranges, validate_dataset
else:
    from .contracts import read_events, read_labels, read_query_ranges, validate_dataset


def validate_directory(directory: Path) -> dict[str, int | bool]:
    event_header, events = read_events(directory / "events.bin")
    label_header, labels = read_labels(directory / "labels.bin")
    range_header, ranges = read_query_ranges(directory / "query_ranges.bin")
    if not (event_header.identity == label_header.identity == range_header.identity):
        raise ValueError("dataset identity mismatch")
    validate_dataset(events, labels, ranges)
    return {"valid": True, "event_count": len(events), "label_count": len(labels),
            "query_count": len(ranges)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a captured UQ dataset")
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(validate_directory(Path(args.dataset))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
