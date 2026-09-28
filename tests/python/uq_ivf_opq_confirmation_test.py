"""Regression checks for explicit cases and adaptive OPQ batch selection."""
import copy
import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.edge_estimation.ivf_experiments import expanded_cases
from scripts.edge_estimation.ivf_opq_confirmation import confirmation_config, select_batch_config

ROOT = Path(__file__).resolve().parents[2]


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT/"configs/edge_estimation/ivf_k256_controlled_full.json").read_text())

    def test_legacy_grid_stays_161(self):
        cases = expanded_cases(self.base)
        self.assertEqual(len(cases), 161)
        self.assertEqual(sum(c["method"] == "hnsw" for c in cases), 7)

    def test_symmetric_beta_and_repeats(self):
        cfg = confirmation_config(self.base)
        rows = expanded_cases(cfg)
        self.assertEqual(len(rows), 189)
        for repeat in range(3):
            a = {(c["beta"],c["ef_search"],c["batch_size"]) for c in rows if c["method"]=="opq" and c["repeat_id"]==repeat}
            b = {(c["beta"],c["ef_search"],c["batch_size"]) for c in rows if c["method"]=="ivf_opq" and c["repeat_id"]==repeat}
            self.assertEqual(a,b)
            self.assertEqual(len(a),27)

    def test_duplicate_rejected(self):
        cfg = confirmation_config(self.base)
        cfg["cases"].append(copy.deepcopy(cfg["cases"][0]))
        with self.assertRaises(ValueError):
            expanded_cases(cfg)

    def test_selection_excludes_fast_below_target_and_marks_fallback(self):
        rows=[
            dict(method="opq",beta="1.2",ef_search="450",recall_at_k=".949",qps="999"),
            dict(method="opq",beta="1.3",ef_search="400",recall_at_k=".951",qps="300"),
            dict(method="opq",beta="1.2",ef_search="700",recall_at_k=".974",qps="200"),
            dict(method="ivf_opq",beta="1.2",ef_search="1075",recall_at_k=".981",qps="250"),
        ]
        cfg, chosen = select_batch_config(self.base, rows)
        s=next(r for r in chosen if r["method"]=="opq" and r["target"]==.95)
        self.assertEqual(s["ef_search"],400)
        s=next(r for r in chosen if r["method"]=="opq" and r["target"]==.97)
        self.assertEqual(s["status"],"OUTSIDE_RECALL_WINDOW")
        s=next(r for r in chosen if r["method"]=="opq" and r["target"]==.98)
        self.assertEqual(s["status"],"NOT_REACHED")
        # IVF picks the same point for multiple anchors; run it only once per batch.
        self.assertEqual(len([c for c in cfg["cases"] if c["method"]=="ivf_opq"]),4)
        self.assertEqual({c["batch_size"] for c in cfg["cases"]},{1,32,128,256})

    def test_explicit_grid_has_no_cartesian_cross_terms(self):
        cfg=confirmation_config(self.base)
        rows=expanded_cases(cfg)
        self.assertFalse(any(c["method"]=="opq" and c["beta"]==1.3 and c["ef_search"]==1075 for c in rows))


if __name__ == "__main__":
    unittest.main()
