#!/usr/bin/env python3
"""Run one comparable end-to-end Recall-QPS grid across HNSW/PQ/QJL/OPQ."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.edge_estimation.final_study.common import seal, unseal, identity, verify_file
from scripts.edge_estimation.final_study.query_contract import evaluate_results
from scripts.edge_estimation.final_study.environment import source_snapshot


METHOD_ORDER = ("hnsw", "pq8", "pq_qjl", "opq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--v0-runner", required=True, type=Path)
    parser.add_argument("--opq-runner", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--pq-sidecar", required=True, type=Path)
    parser.add_argument("--qjl-companion", required=True, type=Path)
    parser.add_argument("--opq-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("dimension", "k", "query_start", "query_count", "warmup_queries", "repeats", "methods"):
        if key not in data:
            raise ValueError(f"configuration is missing {key}")
    if tuple(data["methods"].keys()) != METHOD_ORDER:
        raise ValueError(f"methods must be ordered exactly as {METHOD_ORDER}")
    if any(int(data[key]) <= 0 for key in ("dimension", "k", "query_count", "repeats")):
        raise ValueError("dimension, k, query_count and repeats must be positive")
    for method, grid in data["methods"].items():
        if not grid.get("ef_search") or any(int(value) <= 0 for value in grid["ef_search"]):
            raise ValueError(f"{method} has an invalid ef_search grid")
        betas = grid.get("beta", [None])
        if method == "hnsw" and betas != [None]:
            raise ValueError("HNSW must not define beta")
        if method != "hnsw" and (not betas or any(float(value) <= 0 for value in betas)):
            raise ValueError(f"{method} has an invalid beta grid")
    return data


def cases(config: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for method in METHOD_ORDER:
        grid = config["methods"][method]
        for beta in grid.get("beta", [None]):
            for ef_search in grid["ef_search"]:
                suffix = f"ef{int(ef_search):04d}"
                if beta is not None:
                    suffix = f"b{float(beta):.4f}_".replace(".", "p") + suffix
                yield {"method": method, "beta": beta, "ef_search": int(ef_search),
                       "case_id": f"{method}-{suffix}"}


def read_ground_truth(path: Path, query_start: int, query_count: int, k: int) -> dict[int, set[int]]:
    with path.open("rb") as source:
        raw = source.read(4)
        if len(raw) != 4:
            raise ValueError("empty ground-truth ivecs")
        width = struct.unpack("<i", raw)[0]
        if width < k:
            raise ValueError(f"ground truth width {width} is smaller than k={k}")
        row_bytes = 4 + 4 * width
        source.seek(query_start * row_bytes)
        result: dict[int, set[int]] = {}
        for offset in range(query_count):
            header = source.read(4)
            values = source.read(4 * width)
            if len(header) != 4 or len(values) != 4 * width:
                raise ValueError("ground-truth query range is truncated")
            stored = struct.unpack("<i", header)[0]
            if stored != width:
                raise ValueError("variable-width ivecs is unsupported")
            labels = struct.unpack(f"<{width}i", values)
            result[query_start + offset] = set(labels[:k])
    return result


def recall_at_k(result_path: Path, truth: dict[int, set[int]], k: int) -> float:
    with result_path.open(newline="", encoding="utf-8") as source:
        repeats = 1 + max((int(row.get("repeat_id",0)) for row in csv.DictReader(source)), default=-1)
    report = evaluate_results(result_path, truth, k, repeats)
    if not report["results_stable_across_repeats"]:
        raise ValueError("repeated top-k results differ")
    return report["recall_at_k"]


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    identity: dict[str, Any] = {"path": str(path.resolve()), "size": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns}
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    identity["sha256"] = digest.hexdigest()
    return identity


def command_for(case: dict[str, Any], args: argparse.Namespace,
                config: dict[str, Any], directory: Path) -> list[str]:
    common = ["--index-path", str(args.index), "--query-path", str(args.queries),
              "--dimension", str(config["dimension"]), "--query-start", str(config["query_start"]),
              "--query-count", str(config["query_count"]), "--k", str(config["k"]),
              "--ef-search", str(case["ef_search"]), "--warmup-queries", str(config["warmup_queries"]),
              "--repeats", str(config["repeats"]), "--output", str(directory / "performance.json"),
              "--latency-records-output", str(directory / "latency.csv"),
              "--result-records-output", str(directory / "results.csv")]
    method = case["method"]
    if method == "opq":
        return [str(args.opq_runner), *common, "--artifact-path", str(args.opq_artifact),
                "--beta", str(case["beta"]), "--batch-size", str(config.get("opq_batch_size", 128))]
    command = [str(args.v0_runner), *common]
    if method == "hnsw":
        return [*command, "--method", "baseline"]
    if method == "pq8":
        return [*command, "--method", "approx-no-retry", "--sidecar-path", str(args.pq_sidecar),
                "--beta", str(case["beta"])]
    return [*command, "--method", "residual-threshold", "--sidecar-path", str(args.pq_sidecar),
            "--companion-path", str(args.qjl_companion), "--theta", str(case["beta"])]


def write_points(path: Path, points: list[dict[str, Any]]) -> None:
    fields = ["case_id", "method", "beta", "ef_search", "recall_at_k", "qps",
              "latency_p50_ns", "latency_p95_ns", "latency_p99_ns",
              "batch_prepare_ns_per_query", "attempted_estimates", "pruned_estimates",
              "fallback_estimates"]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: point.get(name, "") for name in fields} for point in points)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    required = [args.v0_runner, args.opq_runner, args.index, args.queries,
                args.ground_truth, args.pq_sidecar, args.qjl_companion, args.opq_artifact]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required paths: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    truth = read_ground_truth(args.ground_truth, int(config["query_start"]),
                              int(config["query_count"]), int(config["k"]))
    expanded = list(cases(config))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_identity": file_identity(args.config),
        "inputs": {name: file_identity(path) if path.is_file() else {"path": str(path.resolve())}
                   for name, path in (("index", args.index), ("queries", args.queries),
                                      ("ground_truth", args.ground_truth), ("pq_sidecar", args.pq_sidecar),
                                      ("qjl_companion", args.qjl_companion),
                                      ("v0_runner", args.v0_runner), ("opq_runner", args.opq_runner),
                                      ("opq_native_cfg", args.opq_artifact / "native.cfg"))},
        "cases": expanded,
        "source": source_snapshot(),
        "opq_artifact_files": [file_identity(p) for p in sorted(args.opq_artifact.rglob("*")) if p.is_file()],
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume or any(previous.get(k)!=v for k,v in manifest.items() if k!="created_at"):
            raise ValueError("output exists or resume inputs/source differ; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        for case in expanded:
            print(" ".join(command_for(case, args, config, args.output / "cases" / case["case_id"])))
        return 0

    points: list[dict[str, Any]] = []
    for number, case in enumerate(expanded, 1):
        directory = args.output / "cases" / case["case_id"]
        directory.mkdir(parents=True, exist_ok=True)
        performance_path = directory / "performance.json"
        result_path = directory / "results.csv"
        command = command_for(case, args, config, directory)
        print(f"[{number}/{len(expanded)}] {case['case_id']}", flush=True)
        case_complete = directory / "complete.json"
        expected_identity = {"manifest":identity(manifest_path),"command":command}
        if args.resume and case_complete.exists():
            previous = unseal(case_complete)
            if previous["identity"] != expected_identity:
                raise ValueError("resume case identity mismatch")
            for entry in previous["outputs"]: verify_file(entry)
        else:
            if performance_path.exists() or result_path.exists():
                raise ValueError("incomplete/unsealed case cannot be resumed; preserve and start a new run")
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, check=False)
            (directory / "runner.log").write_text(completed.stdout, encoding="utf-8")
            (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"{case['case_id']} failed with exit code {completed.returncode}; see {directory / 'runner.log'}")
        performance = json.loads(performance_path.read_text(encoding="utf-8"))
        point = {**case, "recall_at_k": recall_at_k(result_path, truth, int(config["k"])),
                 **{key: performance.get(key) for key in ("qps", "latency_p50_ns", "latency_p95_ns",
                                                          "latency_p99_ns", "batch_prepare_ns_per_query",
                                                          "attempted_estimates", "pruned_estimates",
                                                          "fallback_estimates")}}
        points.append(point)
        if not case_complete.exists():
            seal(case_complete,{"identity":expected_identity,"outputs":[identity(performance_path),identity(result_path)]})
        write_points(args.output / "points.csv", points)
    summary = {"schema_version": 1, "recall_k": config["k"], "point_count": len(points), "points": points}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output / "complete.json").write_text(json.dumps({"valid": True, "point_count": len(points)}, indent=2) + "\n", encoding="utf-8")
    print(f"RECALL_QPS_GRID_COMPLETE points={len(points)} output={args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # concise batch-job failure marker
        print(f"RECALL_QPS_GRID_FAILED: {error}", file=sys.stderr)
        raise SystemExit(2)
