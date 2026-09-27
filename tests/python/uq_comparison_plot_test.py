from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.plot_comparison_results import (  # noqa: E402
    METHODS, generate)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ComparisonPlotTest(unittest.TestCase):
    def test_generates_vector_raster_and_table_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uq-plot-") as temporary:
            data = Path(temporary) / "data"
            data.mkdir()
            quality = []
            operating = []
            timing = []
            timing_raw = []
            for index, method in enumerate(METHODS):
                for alpha, correct, p95 in (
                        (1.0, 0.65 - index * 0.03, 0.02),
                        (1.5, 0.45 - index * 0.03, 0.005)):
                    quality.append({
                        "method": method, "alpha": alpha,
                        "correct_prune_rate": correct,
                        "p95_query_false_prune_rate": p95,
                    })
                for profile, alpha, correct, global_fp, p95, gate_g, gate_p in (
                        ("balanced", 1.0, 0.65 - index * 0.03,
                         0.003, 0.008, 0.005, 0.01),
                        ("strict", 1.5, 0.45 - index * 0.03,
                         0.0005, 0.002, 0.001, 0.0025)):
                    operating.append({
                        "method": method, "profile": profile, "alpha": alpha,
                        "correct_prune_rate": correct,
                        "global_false_prune_rate": global_fp,
                        "p95_query_false_prune_rate": p95,
                        "prune_precision": 0.99,
                        "prunable_coverage": correct + 0.1,
                        "max_global_false_prune_rate": gate_g,
                        "max_p95_query_false_prune_rate": gate_p,
                    })
                median = 100.0 * (index + 1)
                timing.append({
                    "method": method,
                    "execution_mode": "batch" if method in {"opq", "jq"}
                    else "scalar",
                    "query_batch_size": 128 if method in {"opq", "jq"} else 1,
                    "median_elapsed_ms": median,
                    "ns_per_eligible_edge": median,
                    "paired_speed_ratio_vs_reference": 1.0 / (index + 1),
                    "block_min_ms": median * 0.99,
                    "block_max_ms": median * 1.01,
                    "block_cv_percent": 0.5,
                    "backend_mib": 500.0 + index * 10,
                })
                timing_raw.append({"method": method, "elapsed_ns": median * 1e6})
            batch = []
            parity = []
            for batch_size in (1, 8, 32, 128, 600):
                for method_index, method in enumerate(("opq", "jq")):
                    batch.append({
                        "batch_size": batch_size, "method": method,
                        "median_elapsed_ms": 300 / math.sqrt(batch_size) + method_index,
                        "ns_per_query": 1000 / math.sqrt(batch_size),
                        "speedup_vs_batch1": math.sqrt(batch_size),
                    })
                    parity.append({"method": method, "batch_size": batch_size,
                                   "valid": True})
            write_csv(data / "quality_sweep.csv", quality)
            write_csv(data / "operating_points.csv", operating)
            write_csv(data / "timing_summary.csv", timing)
            write_csv(data / "timing_raw.csv", timing_raw)
            write_csv(data / "batch_scaling.csv", batch)
            write_csv(data / "batch_parity.csv", parity)
            output = Path(temporary) / "publication"
            generate(data, output)
            self.assertTrue(
                (output / "figures" / "fig01_quality_cost_pareto.pdf").is_file())
            self.assertTrue(
                (output / "figures" / "fig03_batch_scaling.png").is_file())
            self.assertTrue(
                (output / "tables" / "table1_operating_points.tex").is_file())
            self.assertTrue((output / "complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
