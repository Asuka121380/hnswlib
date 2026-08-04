#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from compare_ratio_search_runs import aggregate, load_runs, main, validate_fresh_queries


class CompareRunsTest(unittest.TestCase):
    def make_run(self, root: Path, name="run") -> Path:
        run = root / name; run.mkdir()
        (run / "complete.json").write_text('{"status":"complete"}')
        metadata = {
            "dataset":"x", "dataset_config_sha256":"a", "query_dataset_sha256":"b",
            "ground_truth_sha256":"c", "index_sha256":"d", "sidecar_sha256":"e",
            "dimension":2, "n_base":10, "k":1, "mode":"ratio-prune", "run_id":name,
            "ef_search":100, "ratio_calibrator_id":"balanced", "ratio_trusted":True,
            "ratio_diagnostic_only":False,
        }
        (run / "metadata.json").write_text(json.dumps(metadata))
        summary = {
            "status":"valid", "query_count":2, "mean_baseline_recall_at_k":1.0,
            "mean_v0_recall_at_k":1.0, "mean_baseline_recall_at_1":1.0,
            "mean_v0_recall_at_1":1.0, "recall_at_k_loss_percentage_points":0.0,
            "worst_v0_recall_at_k":1.0, "mismatch_query_fraction":0.0,
            "baseline_latency_ns":200, "v0_latency_ns":100,
            "peak_rss_bytes":1024,
            "visited_nodes":100, "candidate_expansions":10,
            "ratio_exact_distance_saved":10,
        }
        (run / "summary.json").write_text(json.dumps(summary))
        pd.DataFrame({"query_id":[10,11], "baseline_recall_at_k":[1,1],
                      "v0_recall_at_k":[1,1], "baseline_latency_ns":[100,100],
                      "v0_latency_ns":[50,50]}).to_csv(
                          run / "query_metrics.csv", index=False)
        return run

    def test_load_and_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            runs, _, _ = load_runs([self.make_run(Path(temp))])
            result = aggregate(runs, .1, 5, 5)
            self.assertTrue(bool(result.loc[0, "scientific_gate_pass"]))

    def test_fresh_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); run = self.make_run(temp)
            _, _, frames = load_runs([run])
            manifest = temp / "fresh.json"
            manifest.write_text(json.dumps({"query_ids":[10,11], "phase3_query_ids":[11]}))
            with self.assertRaises(ValueError):
                validate_fresh_queries(frames, manifest)

    def test_smoke_writes_planned_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp); run = self.make_run(temp, "measured")
            output = temp / "phase4"
            old_argv = sys.argv
            try:
                sys.argv = ["compare", "--run-dir", str(run),
                            "--output-dir", str(output), "--mode", "smoke",
                            "--expected-repetitions", "1", "--expected-ef-search", "100",
                            "--expected-operating-points", "balanced"]
                self.assertEqual(main(), 0)
            finally:
                sys.argv = old_argv
            for relative in (
                "tables/recall_performance_frontier.csv",
                "tables/per_query_quality.csv", "tables/work_counters.csv",
                "figures/recall_vs_qps.png",
                "figures/recall_loss_vs_distance_saved.png",
                "figures/latency_distribution.png",
                "figures/per_query_recall_loss.png",
                "PHASE4_HNSW_PROBABILISTIC_PRUNING_REPORT.md",
                "FINAL_GO_NO_GO_DECISION.md", "decision.json"):
                self.assertTrue((output / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
