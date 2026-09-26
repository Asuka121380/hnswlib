from __future__ import annotations

import argparse
import importlib.util
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
            "codec": {"kind": "pq_packed", "m": 2, "nbits": 4,
                      "layout": "packed4"},
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
            args.runner, "quality", str(dataset / "events.bin"),
            str(dataset / "labels.bin"), str(artifact),
            str(fixture / "queries.fvecs"), "1.0"])
        timing = run_json([
            args.runner, "bench-artifact", str(dataset / "events.bin"),
            str(artifact), str(fixture / "queries.fvecs"), "2"])
        if not validation["valid"] or not artifact_validation["valid"]:
            raise RuntimeError("native validation failed")
        if quality["valid_estimate_count"] <= 0 or quality["fallback_count"] != 0:
            raise RuntimeError("quality fixture did not exercise valid packed-PQ scores")
        if (not quality.get("per_query") or
                quality.get("error_percentiles", {}).get("overestimate", {}).get("count", 0) <= 0):
            raise RuntimeError("native quality omitted per-query one-sided error statistics")
        if timing["eligible_events"] != quality["decision_count_s"]:
            raise RuntimeError("quality and timing did not use the same eligible work")
        if timing["source_setup_calls"] <= 0 or timing["query_count"] != 3:
            raise RuntimeError("timing lifecycle counters are incomplete")
        if timing["quality_valid"] or timing["formal_validation_passed"]:
            raise RuntimeError("timing incorrectly claimed independent quality validation")
        orchestrated_validation = root / "orchestrated-validation"
        orchestrated_quality = root / "orchestrated-quality"
        orchestrated_timing = root / "orchestrated-timing"
        subprocess.run([
            args.python, args.run_script, "validate", "--runner", args.runner,
            "--events", str(dataset / "events.bin"), "--queries",
            str(fixture / "queries.fvecs"), "--artifact", str(artifact),
            "--out", str(orchestrated_validation)], check=True)
        subprocess.run([
            args.python, args.run_script, "quality", "--runner", args.runner,
            "--events", str(dataset / "events.bin"), "--labels",
            str(dataset / "labels.bin"), "--queries", str(fixture / "queries.fvecs"),
            "--artifact", str(artifact), "--alpha", "1.0",
            "--out", str(orchestrated_quality)], check=True)
        subprocess.run([
            args.python, args.run_script, "bench", "--runner", args.runner,
            "--events", str(dataset / "events.bin"), "--queries",
            str(fixture / "queries.fvecs"), "--artifact", str(artifact),
            "--validation", str(orchestrated_validation / "validation.json"),
            "--quality-report", str(orchestrated_quality / "quality.json"),
            "--formal", "--repeats", "1", "--out", str(orchestrated_timing)],
            check=True)
        admitted = json.loads((orchestrated_timing / "result.json").read_text(
            encoding="utf-8"))
        if (not admitted["evidence"]["formal_admitted"] or
                not admitted["quality_valid"] or
                not admitted["formal_validation_passed"]):
            raise RuntimeError("formal timing did not retain verified quality evidence")

        opq_artifact = root / "opq-artifact"
        if importlib.util.find_spec("faiss") is not None:
            faiss_pq_artifact = root / "faiss-pq-artifact"
            faiss_pq_config = root / "faiss-pq-config.json"
            faiss_pq_config.write_text(json.dumps({
                "schema_version": 1, "run_id": "uq-faiss-pq-e2e-fixture",
                "representation": {"kind": "direct_unit_edge"},
                "codec": {"kind": "pq_packed", "m": 2, "nbits": 8,
                          "layout": "packed_nbits"},
                "correction": {"kind": "none"},
                "policy": {"kind": "scaled_threshold", "alphas": [1.0]},
                "capture": {"dimension": 8, "k": 10, "ef": 40},
                "trainer": {"provider": "faiss", "seed": 77, "threads": 1,
                            "iterations": 3, "sample_cap": 256,
                            "encode_batch_size": 64}
            }) + "\n", encoding="utf-8")
            subprocess.run([
                args.python, args.train_script, "--config", str(faiss_pq_config),
                "--assets", str(assets), "--catalog", str(catalog),
                "--out", str(faiss_pq_artifact)], check=True)
            faiss_pq_validation = run_json([
                args.runner, "validate-artifact", str(faiss_pq_artifact),
                str(dataset / "events.bin"), str(fixture / "queries.fvecs")])
            if faiss_pq_validation["backend"] != "pq_packed":
                raise RuntimeError("Faiss PQ artifact did not use the packed-PQ runtime")
            opq_config = root / "opq-config.json"
            opq_config.write_text(json.dumps({
                "schema_version": 1, "run_id": "uq-opq-e2e-fixture",
                "representation": {"kind": "rotated_unit_edge"},
                "codec": {"kind": "opq", "m": 2, "nbits": 2,
                          "layout": "packed_nbits"},
                "correction": {"kind": "none"},
                "policy": {"kind": "scaled_threshold", "alphas": [1.0]},
                "capture": {"dimension": 8, "k": 10, "ef": 40},
                "trainer": {"provider": "faiss", "seed": 77, "threads": 1,
                            "iterations": 3, "initial_pq_iterations": 3,
                            "outer_iterations": 2, "sample_cap": 256,
                            "encode_batch_size": 64}
            }) + "\n", encoding="utf-8")
            subprocess.run([
                args.python, args.train_script, "--config", str(opq_config),
                "--assets", str(assets), "--catalog", str(catalog),
                "--out", str(opq_artifact)], check=True)
            opq_validation = run_json([
                args.runner, "validate-artifact", str(opq_artifact),
                str(dataset / "events.bin"), str(fixture / "queries.fvecs")])
            opq_quality = run_json([
                args.runner, "quality", str(dataset / "events.bin"),
                str(dataset / "labels.bin"), str(opq_artifact),
                str(fixture / "queries.fvecs"), "1.0"])
            opq_timing = run_json([
                args.runner, "bench-artifact", str(dataset / "events.bin"),
                str(opq_artifact), str(fixture / "queries.fvecs"), "1"])
            if (opq_validation["backend"] != "opq" or
                    opq_quality["valid_estimate_count"] <= 0 or
                    opq_timing["backend"] != "opq"):
                raise RuntimeError("OPQ fixture did not complete the unified native path")
            prq_artifact = root / "prq-artifact"
            prq_config = root / "prq-config.json"
            prq_config.write_text(json.dumps({
                "schema_version": 1, "run_id": "uq-prq-e2e-fixture",
                "representation": {"kind": "direct_unit_edge"},
                "codec": {"kind": "prq", "nsplits": 2,
                          "stages_per_split": 2, "nbits": 2},
                "correction": {"kind": "none"},
                "policy": {"kind": "scaled_threshold", "alphas": [1.0]},
                "capture": {"dimension": 8, "k": 10, "ef": 40},
                "trainer": {"provider": "faiss", "seed": 77, "threads": 1,
                            "iterations": 3, "beam_size": 3,
                            "sample_cap": 256, "encode_batch_size": 64}
            }) + "\n", encoding="utf-8")
            subprocess.run([
                args.python, args.train_script, "--config", str(prq_config),
                "--assets", str(assets), "--catalog", str(catalog),
                "--out", str(prq_artifact)], check=True)
            prq_validation = run_json([
                args.runner, "validate-artifact", str(prq_artifact),
                str(dataset / "events.bin"), str(fixture / "queries.fvecs")])
            prq_quality = run_json([
                args.runner, "quality", str(dataset / "events.bin"),
                str(dataset / "labels.bin"), str(prq_artifact),
                str(fixture / "queries.fvecs"), "1.0"])
            if (prq_validation["backend"] != "prq" or
                    prq_quality["valid_estimate_count"] <= 0):
                raise RuntimeError("PRQ fixture did not complete the unified native path")
            jq_artifact = root / "jq-artifact"
            jq_config = root / "jq-config.json"
            jq_config.write_text(json.dumps({
                "schema_version": 1, "run_id": "uq-jq-e2e-fixture",
                "representation": {"kind": "rotated_unit_edge"},
                "codec": {"kind": "jq", "m": 2, "nbits": 8},
                "correction": {"kind": "none"},
                "policy": {"kind": "scaled_threshold", "alphas": [1.0]},
                "capture": {"dimension": 8, "k": 10, "ef": 40},
                "trainer": {"provider": "behavioral_port", "rotation_seed": 17,
                            "initializer_seed": 42, "sample_cap": 512,
                            "encode_batch_size": 64}
            }) + "\n", encoding="utf-8")
            subprocess.run([
                args.python, args.train_script, "--config", str(jq_config),
                "--assets", str(assets), "--catalog", str(catalog),
                "--out", str(jq_artifact)], check=True)
            jq_validation = run_json([
                args.runner, "validate-artifact", str(jq_artifact),
                str(dataset / "events.bin"), str(fixture / "queries.fvecs")])
            jq_quality = run_json([
                args.runner, "quality", str(dataset / "events.bin"),
                str(dataset / "labels.bin"), str(jq_artifact),
                str(fixture / "queries.fvecs"), "1.0"])
            if (jq_validation["backend"] != "jq" or
                    jq_quality["valid_estimate_count"] <= 0):
                raise RuntimeError("JQ fixture did not complete the unified native path")
            rabitq_artifact = root / "rabitq-artifact"
            rabitq_config = root / "rabitq-config.json"
            rabitq_config.write_text(json.dumps({
                "schema_version": 1, "run_id": "uq-rabitq-e2e-fixture",
                "representation": {"kind": "rotated_unit_edge"},
                "codec": {"kind": "rabitq", "nbits": 1,
                          "query_bits": 0, "centroid": "zero"},
                "correction": {"kind": "faiss_native_factors"},
                "policy": {"kind": "scaled_threshold", "alphas": [1.0]},
                "capture": {"dimension": 8, "k": 10, "ef": 40},
                "trainer": {"provider": "faiss", "threads": 1,
                            "rotation_seed": 17, "sample_cap": 256,
                            "encode_batch_size": 64}
            }) + "\n", encoding="utf-8")
            subprocess.run([
                args.python, args.train_script, "--config", str(rabitq_config),
                "--assets", str(assets), "--catalog", str(catalog),
                "--out", str(rabitq_artifact)], check=True)
            rabitq_validation = run_json([
                args.runner, "validate-artifact", str(rabitq_artifact),
                str(dataset / "events.bin"), str(fixture / "queries.fvecs")])
            rabitq_quality = run_json([
                args.runner, "quality", str(dataset / "events.bin"),
                str(dataset / "labels.bin"), str(rabitq_artifact),
                str(fixture / "queries.fvecs"), "1.0"])
            rabitq_timing = run_json([
                args.runner, "bench-artifact", str(dataset / "events.bin"),
                str(rabitq_artifact), str(fixture / "queries.fvecs"), "1"])
            if (rabitq_validation["backend"] != "rabitq" or
                    rabitq_quality["valid_estimate_count"] <= 0 or
                    rabitq_timing["backend"] != "rabitq"):
                raise RuntimeError("RaBitQ fixture did not complete the unified native path")
        matrix = root / "matrix.json"
        matrix.write_text(json.dumps({
            "schema_version": 1, "reference": "packed-a", "blocks": 2,
            "repeats": 2, "seed": 77, "isa_profile": "portable",
            "methods": [
                {"name": "packed-a", "artifact": str(artifact),
                 "validation": str(orchestrated_validation / "validation.json"),
                 "quality_report": str(orchestrated_quality / "quality.json")},
                {"name": "packed-b", "artifact": str(artifact),
                 "validation": str(orchestrated_validation / "validation.json"),
                 "quality_report": str(orchestrated_quality / "quality.json")},
            ]
        }) + "\n", encoding="utf-8")
        paired = root / "paired"
        subprocess.run([
            args.python, args.run_script, "bench", "--matrix", str(matrix),
            "--events", str(dataset / "events.bin"),
            "--queries", str(fixture / "queries.fvecs"),
            "--runner", args.runner, "--formal", "--out", str(paired)], check=True)
        paired_result = json.loads((paired / "result.json").read_text(encoding="utf-8"))
        if len(paired_result["raw_records"]) != 8:
            raise RuntimeError("paired timing schedule is incomplete")
        if not paired_result["evidence"]["formal_admitted"]:
            raise RuntimeError("paired timing omitted per-method formal evidence")
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
