from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.edge_estimation.run import convert_split, preflight, wrap_legacy  # noqa: E402
from scripts.edge_estimation.contracts import validate_config_v1  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class RunContractTest(unittest.TestCase):
    def test_all_six_formal_configs_are_complete_schema_v1(self) -> None:
        directory = ROOT / "configs" / "edge_estimation" / "methods"
        names = {path.stem for path in directory.glob("*.json")}
        self.assertEqual(names, {"pq8", "pq4", "opq", "prq", "jq", "rabitq"})
        for path in directory.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            validate_config_v1(value)
            self.assertEqual(value["capture"]["dimension"], 960)

    def test_convert_legacy_split_preserves_identity_and_partition(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uq-split-") as temporary:
            root = Path(temporary)
            source = root / "old.json"
            write_json(source, {"development": [2, 0], "selection": [1], "audit": [3]})
            output = root / "converted"
            convert_split(argparse.Namespace(
                input=str(source), query_count=4, out=str(output)))
            result = json.loads((output / "split.json").read_text(encoding="utf-8"))
            self.assertEqual(result["splits"]["development"], [2, 0])
            self.assertEqual(result["split_counts"],
                             {"development": 2, "selection": 1, "audit": 1})
            self.assertEqual(result["source"]["sha256"],
                             hashlib.sha256(source.read_bytes()).hexdigest())

    def test_preflight_rejects_expected_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uq-preflight-") as temporary:
            root = Path(temporary)
            index = root / "index.bin"
            queries = root / "queries.fvecs"
            index.write_bytes(b"index")
            queries.write_bytes(b"queries")
            assets = root / "assets.json"
            config = root / "config.json"
            write_json(assets, {
                "schema_version": 1,
                "index": {"path": str(index), "sha256": "0" * 64},
                "queries": str(queries),
            })
            write_json(config, {
                "schema_version": 1, "run_id": "test", "representation": {"kind": "x"},
                "codec": {"kind": "x"}, "correction": {"kind": "x"},
                "policy": {"kind": "x", "alphas": [1.0]},
            })
            output = root / "report"
            code = preflight(argparse.Namespace(
                assets=str(assets), config=str(config), out=str(output),
                hash_assets=False, stage="capture", ledger=None))
            self.assertEqual(code, 3)
            report = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "blocked_identity_mismatch")
            self.assertEqual(report["identity_mismatches"], ["index"])

    def test_wrap_legacy_freezes_source_bytes_and_catalog_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uq-wrap-legacy-") as temporary:
            root = Path(temporary)
            catalog = root / "catalog"
            catalog.mkdir()
            write_json(catalog / "manifest.json", {
                "identity_sha256": "1" * 64, "edge_count": 7})
            sidecar = root / "old.v0meta"
            sidecar.write_bytes(b"legacy-sidecar-fixture")
            output = root / "artifact"
            wrap_legacy(argparse.Namespace(
                sidecar=str(sidecar), residual=None, catalog=str(catalog),
                out=str(output), ledger=None))
            native = (output / "native.cfg").read_text(encoding="ascii")
            self.assertIn("format=uq-pq-legacy/1", native)
            self.assertIn("catalog_identity=" + "1" * 64, native)
            self.assertEqual((output / "legacy" / "sidecar.bin").read_bytes(),
                             sidecar.read_bytes())


if __name__ == "__main__":
    unittest.main()
