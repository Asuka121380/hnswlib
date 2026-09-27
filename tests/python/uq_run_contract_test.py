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

from scripts.edge_estimation.run import (  # noqa: E402
    _timing_evidence, convert_split, preflight, wrap_legacy)
from scripts.edge_estimation.contracts import (  # noqa: E402
    file_entry, validate_config_v1)
from scripts.edge_estimation.prepare_timing_matrix import (  # noqa: E402
    METHODS, build_matrix, write_matrix)


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

    def test_quality_v2_identity_is_admitted_for_formal_timing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uq-timing-evidence-") as temporary:
            root = Path(temporary)
            events, queries = root / "events.bin", root / "queries.fvecs"
            events.write_bytes(b"events")
            queries.write_bytes(b"queries")
            artifact = root / "artifact"
            artifact.mkdir()
            (artifact / "manifest.json").write_bytes(b"manifest")
            (artifact / "native.cfg").write_bytes(b"native")

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            identities = {
                "events_sha256": digest(events),
                "queries_sha256": digest(queries),
                "artifact_manifest_sha256": digest(artifact / "manifest.json"),
                "artifact_native_cfg_sha256": digest(artifact / "native.cfg"),
            }
            validation = root / "validation.json"
            write_json(validation, {"valid": True,
                                    "input_identities": identities})
            quality = root / "quality.json"
            write_json(quality, {
                "stage": "quality-sweep", "summaries": [{"alpha": 1.0}],
                "inputs": {
                    "events": {"sha256": identities["events_sha256"]},
                    "queries": {"sha256": identities["queries_sha256"]},
                    "artifact_manifest": {
                        "sha256": identities["artifact_manifest_sha256"]},
                    "artifact_native_config": {
                        "sha256": identities["artifact_native_cfg_sha256"]},
                },
            })
            result = _timing_evidence(
                str(validation), str(quality), str(events), str(queries),
                str(artifact), True)
            self.assertTrue(result["formal_admitted"])
            self.assertIsNotNone(result["quality"])

    def test_build_timing_matrix_freezes_all_six_methods(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uq-timing-matrix-") as temporary:
            root = Path(temporary)
            for method in METHODS:
                artifact = root / f"artifact-{method}"
                artifact.mkdir()
                (artifact / "manifest.json").write_text("{}", encoding="utf-8")
                (artifact / "native.cfg").write_text("format=test\n", encoding="ascii")
                validation = root / f"validation-{method}"
                validation.mkdir()
                write_json(validation / "validation.json", {"valid": True})
                quality = root / f"quality-v2-{method}"
                quality.mkdir()
                quality_report = quality / "quality.json"
                write_json(quality_report, {
                    "stage": "quality-sweep", "summaries": [{"alpha": 1.0}]})
                write_json(quality / "complete.json", {
                    "stage": "quality-sweep",
                    "outputs": {"quality": file_entry(quality_report)},
                })
            matrix = build_matrix(root, blocks=5, repeats=5,
                                  seed=20260924, reference="pq8")
            self.assertEqual([item["name"] for item in matrix["methods"]],
                             list(METHODS))
            batch_matrix = build_matrix(
                root, blocks=3, repeats=3, seed=20260927,
                reference="opq", method_names=("opq", "jq"))
            self.assertEqual([item["name"] for item in batch_matrix["methods"]],
                             ["opq", "jq"])
            with self.assertRaisesRegex(ValueError, "reference must be included"):
                build_matrix(root, blocks=1, repeats=1, seed=1,
                             reference="pq8", method_names=("opq", "jq"))
            output = root / "matrix.json"
            self.assertEqual(write_matrix(output, matrix), "created")
            self.assertEqual(write_matrix(output, matrix), "reused")


if __name__ == "__main__":
    unittest.main()
