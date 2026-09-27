#!/usr/bin/env python3
import csv
import importlib.util
import json
import struct
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "edge_estimation" / "run_recall_qps_grid.py"
SPEC = importlib.util.spec_from_file_location("recall_qps_grid", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> None:
    config = MODULE.load_config(ROOT / "configs" / "edge_estimation" / "recall_qps_grid_v1.json")
    expanded = list(MODULE.cases(config))
    assert len(expanded) == 70, len(expanded)
    assert expanded[0]["method"] == "hnsw"
    assert expanded[-1] == {"method": "opq", "beta": 1.4, "ef_search": 800,
                            "case_id": "opq-b1p4000_ef0800"}

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        ground_truth = root / "gt.ivecs"
        with ground_truth.open("wb") as target:
            for labels in ((1, 2, 3), (4, 5, 6)):
                target.write(struct.pack("<i3i", 3, *labels))
        truth = MODULE.read_ground_truth(ground_truth, 0, 2, 2)
        results = root / "results.csv"
        with results.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(("query_id", "rank", "label", "distance"))
            writer.writerows(((0, 0, 1, 0.1), (0, 1, 9, 0.2),
                              (1, 0, 4, 0.1), (1, 1, 5, 0.2)))
        assert MODULE.recall_at_k(results, truth, 2) == 0.75
    print("UQ_RECALL_QPS_GRID_TEST=PASS")


if __name__ == "__main__":
    main()
