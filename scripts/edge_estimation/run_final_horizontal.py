from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.edge_estimation.contracts import (  # noqa: E402
    atomic_output_dir, file_entry, load_strict_json, sha256_file)
from scripts.edge_estimation.run import _timing_evidence  # noqa: E402
from scripts.edge_estimation.timing import (  # noqa: E402
    paired_schedule, summarize_paired)


METHODS = ("pq8", "pq4", "opq", "prq", "jq", "rabitq")
BATCH_METHODS = frozenset(("opq", "jq"))
BATCH_ENGINE = "blas_sgemv_sgemm"


def benchmark_command(runner: Path, events: Path, artifact: Path,
                      queries: Path, method: str,
                      batch_size: int) -> list[str]:
    common = [str(runner), str(events), str(artifact), str(queries)]
    if method in BATCH_METHODS:
        return [common[0], "bench-batch-artifact", *common[1:],
                str(batch_size), "1"]
    return [common[0], "bench-artifact", *common[1:], "1"]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    matrix = load_strict_json(args.matrix)
    raw_methods = matrix.get("methods")
    if not isinstance(raw_methods, list):
        raise ValueError("timing matrix methods are missing")
    names = [str(item["name"]) for item in raw_methods]
    if tuple(names) != METHODS:
        raise ValueError(
            "final horizontal timing requires canonical six-method order")
    reference = str(matrix.get("reference", "pq8"))
    if reference not in names:
        raise ValueError("timing reference is absent")

    runner = Path(args.runner).resolve()
    events = Path(args.events).resolve()
    queries = Path(args.queries).resolve()
    artifacts = {
        str(item["name"]): Path(str(item["artifact"])).resolve()
        for item in raw_methods
    }
    evidence = {
        str(item["name"]): _timing_evidence(
            item.get("validation"), item.get("quality_report"),
            str(events), str(queries), str(artifacts[str(item["name"])]),
            args.formal)
        for item in raw_methods
    }
    dataset_id = sha256_file(events)
    build_id = sha256_file(runner)
    contract = (
        f"selected-query-preparation:b{args.batch_size}:{BATCH_ENGINE}")
    records: list[dict[str, object]] = []

    for slot in paired_schedule(
            names, int(matrix.get("blocks", 3)),
            int(matrix.get("repeats", 3)),
            int(matrix.get("seed", 20260927))):
        method = str(slot["method"])
        command = benchmark_command(
            runner, events, artifacts[method], queries, method, args.batch_size)
        value = json.loads(subprocess.run(
            command, check=True, text=True, capture_output=True).stdout)
        uses_batch = method in BATCH_METHODS
        if uses_batch and (
                value.get("batch_rotation_engine") != BATCH_ENGINE or
                int(value.get("query_batch_size", 0)) != args.batch_size):
            raise ValueError(f"{method} did not use the requested BLAS batch path")
        records.append({
            **slot,
            "elapsed_ns": value["raw_elapsed_ns"][0],
            "query_count": value["query_count"],
            "eligible_events": value["eligible_events"],
            "dataset_id": dataset_id,
            "build_id": build_id,
            # These normalized fields identify the shared comparison contract.
            "mode": "selected_query_preparation_estimator",
            "thread_count": 1,
            "isa_profile": contract,
            # Preserve the native execution path for auditing.
            "native_mode": value["mode"],
            "execution_mode": "batch" if uses_batch else "scalar",
            "query_batch_size": args.batch_size if uses_batch else 1,
            "batch_rotation_engine": (
                value["batch_rotation_engine"] if uses_batch else None),
            "checksum": value["checksum"],
            "memory_report": value["memory_report"],
            "evidence": evidence[method],
        })

    result = {
        "schema_version": 1,
        "mode": "randomized_paired_selected_query_preparation",
        "execution_contract": {
            method: ({"mode": "batch", "query_batch_size": args.batch_size,
                      "batch_rotation_engine": BATCH_ENGINE}
                     if method in BATCH_METHODS else {"mode": "scalar"})
            for method in names
        },
        "raw_records": records,
        "summary": summarize_paired(records, reference),
        "evidence": {"by_method": evidence,
                     "formal_admitted": bool(args.formal)},
        "input_identities": {
            "events_sha256": dataset_id,
            "runner_sha256": build_id,
            "queries_sha256": sha256_file(queries),
        },
    }
    with atomic_output_dir(args.out) as partial:
        result_path = partial / "result.json"
        _write_json(result_path, result)
        _write_json(partial / "complete.json", {
            "schema_version": 1,
            "stage": "final-horizontal-timing",
            "outputs": [file_entry(result_path)],
        })
    print(json.dumps({"record_count": len(records), "reference": reference,
                      "batch_size": args.batch_size}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one-node six-method timing with batched OPQ/JQ")
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
