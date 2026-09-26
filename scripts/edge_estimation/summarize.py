from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.edge_estimation.contracts import load_strict_json
else:
    from .contracts import load_strict_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine quality and timing reports")
    parser.add_argument("--quality", required=True)
    parser.add_argument("--timing", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    value = {"schema_version": 1,
             "quality": load_strict_json(args.quality),
             "timing": load_strict_json(args.timing),
             "formal_result_eligible": False,
             "limitations": ["Eligibility must be set only after frozen-asset validation."]}
    Path(args.out).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
