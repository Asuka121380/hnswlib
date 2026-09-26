from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_json(command: list[str]) -> dict:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--train-script", required=True)
    parser.add_argument("--run-script", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="uq-e2e-") as temporary:
        root = Path(temporary)
        fixture, catalog, dataset, artifact = (
            root / "fixture", root / "catalog", root / "dataset", root / "artifact")
        fixture.mkdir()
        subprocess.run([args.fixture, str(fixture)], check=True)
        catalog_result = run_json([
            args.capture, "--catalog", "--index", str(fixture / "index.bin"),
            "--dimension", "8", str(catalog)])
        if catalog_result["node_count"] != 96 or catalog_result["edge_count"] <= 0:
            raise RuntimeError("catalog fixture did not contain the expected graph")
        capture_result = run_json([
            args.capture, "--index", str(fixture / "index.bin"),
            "--queries", str(fixture / "queries.fvecs"),
            "--query-ids", str(fixture / "query_ids.txt"),
            "--dimension", "8", "--k", "10", "--ef", "40",
            "--out", str(dataset)])
        if capture_result["query_count"] != 3 or capture_result["event_count"] <= 0:
            raise RuntimeError("capture fixture did not record all selected queries")

        assets = root / "assets.json"
        assets.write_text(json.dumps({
            "schema_version": 1,
            "index": str(fixture / "index.bin"),
            "base": str(fixture / "base.fvecs"),
            "queries": str(fixture / "queries.fvecs")}) + "\n", encoding="utf-8")
        config = root / "config.json"
        config.write_text(json.dumps({
            "schema_version": 1, "run_id": "uq-e2e-fixture",
            "representation": {"kind": "direct_unit_edge"},
            "codec": {"kind": "pq_packed", "m": 2, "nbits": 2,
                      "layout": "packed_nbits"},
            "correction": {"kind": "none"},
            "policy": {"kind": "scaled_threshold", "alphas": [1.0]},
            "capture": {"dimension": 8, "k": 10, "ef": 40},
            "trainer": {"seed": 77, "iterations": 3, "sample_cap": 128}
        }) + "\n", encoding="utf-8")
        subprocess.run([
            args.python, args.train_script, "--config", str(config),
            "--assets", str(assets), "--catalog", str(catalog),
            "--out", str(artifact)], check=True)

        validation = run_json([
            args.runner, "validate", str(dataset / "events.bin"),
            str(dataset / "labels.bin"), str(dataset / "query_ranges.bin")])
        artifact_validation = run_json([
            args.runner, "validate-artifact", str(artifact),
            str(dataset / "events.bin"), str(fixture / "queries.fvecs")])
        quality = run_json([
            args.runner, "quality-pq", str(dataset / "events.bin"),
            str(dataset / "labels.bin"), str(artifact),
            str(fixture / "queries.fvecs"), "1.0"])
        timing = run_json([
            args.runner, "bench-pq", str(dataset / "events.bin"),
            str(artifact), str(fixture / "queries.fvecs"), "2"])
        if not validation["valid"] or not artifact_validation["valid"]:
            raise RuntimeError("native validation failed")
        if quality["valid_estimate_count"] <= 0 or quality["fallback_count"] != 0:
            raise RuntimeError("quality fixture did not exercise valid packed-PQ scores")
        if timing["eligible_events"] != quality["decision_count_s"]:
            raise RuntimeError("quality and timing did not use the same eligible work")
        if timing["source_setup_calls"] <= 0 or timing["query_count"] != 3:
            raise RuntimeError("timing lifecycle counters are incomplete")
        matrix = root / "matrix.json"
        matrix.write_text(json.dumps({
            "schema_version": 1, "reference": "packed-a", "blocks": 2,
            "repeats": 2, "seed": 77, "isa_profile": "portable",
            "methods": [{"name": "packed-a", "artifact": str(artifact)},
                        {"name": "packed-b", "artifact": str(artifact)}]
        }) + "\n", encoding="utf-8")
        paired = root / "paired"
        subprocess.run([
            args.python, args.run_script, "bench", "--matrix", str(matrix),
            "--events", str(dataset / "events.bin"),
            "--queries", str(fixture / "queries.fvecs"),
            "--runner", args.runner, "--out", str(paired)], check=True)
        paired_result = json.loads((paired / "result.json").read_text(encoding="utf-8"))
        if len(paired_result["raw_records"]) != 8:
            raise RuntimeError("paired timing schedule is incomplete")
        corrupt = root / "corrupt-artifact"
        shutil.copytree(artifact, corrupt)
        records = corrupt / "records" / "edges.bin"
        payload = bytearray(records.read_bytes())
        payload[-1] ^= 1
        records.write_bytes(payload)
        rejected = subprocess.run([
            args.runner, "validate-artifact", str(corrupt),
            str(dataset / "events.bin"), str(fixture / "queries.fvecs")],
            text=True, capture_output=True)
        if rejected.returncode == 0 or "hash mismatch" not in rejected.stderr:
            raise RuntimeError("corrupt artifact was not rejected by its hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
