import sys
import unittest
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.import_legacy import import_jsonl, legacy_capabilities  # noqa: E402


class LegacyImportTest(unittest.TestCase):
    def test_sparse_trace_cannot_claim_timing(self):
        self.assertFalse(legacy_capabilities("phase1_p3_events")["ordered_timing"])
        self.assertFalse(legacy_capabilities("PQREPL01")["quality"])

    def test_import_preserves_rows_and_restricts_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.jsonl"
            source.write_text('{"query_id":7,"threshold":3.0}\n', encoding="utf-8")
            output = root / "imported"
            import_jsonl(source, "phase1_p3_events", output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(1, manifest["row_count"])
            self.assertFalse(manifest["capabilities"]["ordered_timing"])
            self.assertEqual([], manifest["fabricated_fields"])


if __name__ == "__main__":
    unittest.main()
