from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.edge_estimation.contracts import (
        append_ledger, atomic_output_dir, file_entry, load_strict_json, read_events,
        read_labels, read_query_ranges, sha256_file, validate_dataset,
    )
    from scripts.edge_estimation.capabilities import environment_capabilities
else:
    from .contracts import (
        append_ledger, atomic_output_dir, file_entry, load_strict_json, read_events,
        read_labels, read_query_ranges, sha256_file, validate_dataset,
    )
    from .capabilities import environment_capabilities

from scripts.edge_estimation.quality import ScoredEvent, summarize as summarize_quality
from scripts.edge_estimation.timing import paired_schedule, summarize_paired


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _complete(directory: Path, stage: str, outputs: Sequence[Path]) -> None:
    _write_json(directory / "complete.json", {
        "schema_version": 1,
        "stage": stage,
        "outputs": [file_entry(path) for path in outputs],
    })


def _require_keys(value: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{name} missing required keys: {', '.join(missing)}")


def preflight(args: argparse.Namespace) -> int:
    assets = load_strict_json(args.assets)
    config = load_strict_json(args.config)
    _require_keys(assets, ["schema_version", "index", "queries"], "assets")
    _require_keys(config, ["schema_version", "representation", "codec",
                           "correction", "policy"], "config")
    if assets["schema_version"] != 1 or config["schema_version"] != 1:
        raise ValueError("only schema_version=1 is supported")
    asset_status: dict[str, object] = {}
    for key, raw_path in assets.items():
        if key == "schema_version":
            continue
        if not isinstance(raw_path, str):
            raise ValueError(f"asset path {key} must be a string")
        path = Path(raw_path).expanduser().resolve()
        asset_status[key] = {"path": str(path), "exists": path.is_file(),
                             "size": path.stat().st_size if path.is_file() else None}
        if args.hash_assets and path.is_file():
            asset_status[key]["sha256"] = sha256_file(path)
    missing = [key for key, value in asset_status.items() if not value["exists"]]
    report = {
        "schema_version": 1,
        "stage": "preflight",
        "status": "blocked_missing_assets" if missing else "ready",
        "missing_assets": missing,
        "formal_asset_identity_complete": not missing and args.hash_assets,
        "assets": asset_status,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "cpu_count": os.cpu_count(),
                        "capabilities": environment_capabilities()},
        "resolved_config": config,
    }
    with atomic_output_dir(args.out) as partial:
        report_path = partial / "audit.json"
        _write_json(report_path, report)
        _complete(partial, "preflight", [report_path])
    print(json.dumps({"status": report["status"], "out": str(Path(args.out).resolve())}))
    return 0 if not missing else 3


def validate(args: argparse.Namespace) -> int:
    if args.artifact:
        if not args.runner or not args.events or not args.queries:
            raise ValueError("artifact validation requires --runner, --events and --queries")
        command = [str(Path(args.runner).resolve()), "validate-artifact",
                   str(Path(args.artifact).resolve()), str(Path(args.events).resolve()),
                   str(Path(args.queries).resolve())]
        report = json.loads(subprocess.run(command, check=True, text=True,
                                           capture_output=True).stdout)
        with atomic_output_dir(args.out) as partial:
            report_path = partial / "validation.json"
            _write_json(report_path, report)
            _complete(partial, "validate-artifact", [report_path])
        print(json.dumps(report))
        return 0
    if not args.labels or not args.ranges:
        raise ValueError("dataset validation requires --labels and --ranges")
    event_header, events = read_events(args.events)
    label_header, labels = read_labels(args.labels)
    range_header, ranges = read_query_ranges(args.ranges)
    if not (event_header.identity == label_header.identity == range_header.identity and
            event_header.dimension == label_header.dimension == range_header.dimension):
        raise ValueError("dataset identity mismatch")
    validate_dataset(events, labels, ranges)
    report = {"schema_version": 1, "valid": True,
              "event_count": len(events), "label_count": len(labels),
              "query_count": len(ranges), "dimension": event_header.dimension}
    with atomic_output_dir(args.out) as partial:
        report_path = partial / "validation.json"
        _write_json(report_path, report)
        _complete(partial, "validate", [report_path])
    print(json.dumps(report))
    return 0


def bench(args: argparse.Namespace) -> int:
    if args.matrix:
        if not args.queries:
            raise ValueError("matrix benchmark requires --queries")
        matrix = load_strict_json(args.matrix)
        methods = matrix.get("methods")
        if (matrix.get("schema_version") != 1 or not isinstance(methods, list) or
                not methods):
            raise ValueError("invalid timing matrix")
        names = [str(item["name"]) for item in methods]
        artifacts = {str(item["name"]): str(Path(item["artifact"]).resolve())
                     for item in methods}
        blocks = int(matrix.get("blocks", 5))
        repeats = int(matrix.get("repeats", 5))
        seed = int(matrix.get("seed", 20260924))
        reference = str(matrix.get("reference", names[0]))
        dataset_id = sha256_file(args.events)
        build_id = sha256_file(args.runner)
        records = []
        for slot in paired_schedule(names, blocks, repeats, seed):
            command = [str(Path(args.runner).resolve()), "bench-pq",
                       str(Path(args.events).resolve()), artifacts[str(slot["method"])],
                       str(Path(args.queries).resolve()), "1"]
            value = json.loads(subprocess.run(command, check=True, text=True,
                                              capture_output=True).stdout)
            records.append({**slot, "elapsed_ns": value["raw_elapsed_ns"][0],
                            "query_count": value["query_count"],
                            "eligible_events": value["eligible_events"],
                            "dataset_id": dataset_id, "build_id": build_id,
                            "mode": value["mode"], "thread_count": 1,
                            "isa_profile": matrix.get("isa_profile", "portable"),
                            "checksum": value["checksum"],
                            "memory_report": value["memory_report"]})
        result = {"schema_version": 1, "mode": "randomized_paired_blocks",
                  "raw_records": records,
                  "summary": summarize_paired(records, reference)}
        with atomic_output_dir(args.out) as partial:
            result_path = partial / "result.json"
            _write_json(result_path, result)
            _complete(partial, "bench", [result_path])
        print(json.dumps({"record_count": len(records), "reference": reference}))
        return 0
    if args.artifact:
        if not args.queries:
            raise ValueError("artifact benchmark requires --queries")
        command = [str(Path(args.runner).resolve()), "bench-pq",
                   str(Path(args.events).resolve()), str(Path(args.artifact).resolve()),
                   str(Path(args.queries).resolve()), str(args.repeats)]
    else:
        command = [str(Path(args.runner).resolve()), "bench",
                   str(Path(args.events).resolve()), str(args.repeats)]
    completed = subprocess.run(command, check=True, text=True,
                               capture_output=True)
    result = json.loads(completed.stdout)
    with atomic_output_dir(args.out) as partial:
        result_path = partial / "result.json"
        _write_json(result_path, result)
        _complete(partial, "bench", [result_path])
    print(json.dumps(result))
    return 0


def capture(args: argparse.Namespace) -> int:
    assets = load_strict_json(args.assets)
    config = load_strict_json(args.config)
    split = load_strict_json(args.split)
    capture_config = config.get("capture", {})
    query_ids = split.get("query_ids")
    if query_ids is None and args.split_name:
        query_ids = split.get("splits", {}).get(args.split_name)
    if not isinstance(query_ids, list) or not query_ids or not all(
            isinstance(value, int) and value >= 0 for value in query_ids):
        raise ValueError("split must provide a non-empty list of original query_ids")
    dimension = int(capture_config.get("dimension", args.dimension or 0))
    k = int(capture_config.get("k", args.k or 0))
    ef = int(capture_config.get("ef", args.ef or 0))
    if not dimension or not k or ef < k:
        raise ValueError("capture requires dimension, k and ef>=k")
    with tempfile.TemporaryDirectory(prefix="uq-query-ids-") as temporary:
        ids_path = Path(temporary) / "query_ids.txt"
        ids_path.write_text("".join(f"{value}\n" for value in query_ids), encoding="ascii")
        command = [str(Path(args.runner).resolve()),
                   "--index", str(Path(assets["index"]).resolve()),
                   "--queries", str(Path(assets["queries"]).resolve()),
                   "--query-ids", str(ids_path), "--dimension", str(dimension),
                   "--k", str(k), "--ef", str(ef),
                   "--out", str(Path(args.out).resolve())]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
    print(completed.stdout.strip())
    return 0


def quality(args: argparse.Namespace) -> int:
    if args.runner:
        required = (args.events, args.labels, args.artifact, args.queries)
        if any(value is None for value in required):
            raise ValueError("native quality requires events, labels, artifact and queries")
        results = []
        for alpha in args.alpha:
            command = [str(Path(args.runner).resolve()), "quality-pq",
                       str(Path(args.events).resolve()), str(Path(args.labels).resolve()),
                       str(Path(args.artifact).resolve()), str(Path(args.queries).resolve()), str(alpha)]
            results.append(json.loads(subprocess.run(command, check=True, text=True,
                                                     capture_output=True).stdout))
        report = {"schema_version": 1, "native": True, "summaries": results}
        with atomic_output_dir(args.out) as partial:
            report_path = partial / "quality.json"; _write_json(report_path, report)
            _complete(partial, "quality", [report_path])
        print(json.dumps({"alphas": args.alpha, "native": True}))
        return 0
    if not args.scores:
        raise ValueError("quality requires --scores or --runner")
    rows = []
    with Path(args.scores).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            try:
                rows.append(ScoredEvent(**value))
            except TypeError as error:
                raise ValueError(f"invalid score row {line_number}: {error}") from error
    alphas = [float(value) for value in args.alpha]
    report = {"schema_version": 1,
              "summaries": [summarize_quality(rows, alpha) for alpha in alphas]}
    with atomic_output_dir(args.out) as partial:
        report_path = partial / "quality.json"
        _write_json(report_path, report)
        _complete(partial, "quality", [report_path])
    print(json.dumps({"event_count": len(rows), "alphas": alphas}))
    return 0


def catalog(args: argparse.Namespace) -> int:
    assets = load_strict_json(args.assets)
    command = [str(Path(args.runner).resolve()), "--catalog", "--index",
               str(Path(assets["index"]).resolve()), "--dimension", str(args.dimension),
               str(Path(args.out).resolve())]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    print(completed.stdout.strip()); return 0


def train_encode_command(args: argparse.Namespace) -> int:
    script = Path(__file__).with_name("train_encode.py")
    command = [args.trainer_python or sys.executable, str(script), "--config", args.config,
               "--assets", args.assets, "--catalog", args.catalog, "--out", args.out]
    subprocess.run(command, check=True); return 0


def summarize_outputs(args: argparse.Namespace) -> int:
    quality_value = load_strict_json(args.quality) if args.quality else None
    timing_value = load_strict_json(args.timing) if args.timing else None
    report = {"schema_version": 1, "quality": quality_value,
              "timing": timing_value,
              "limitations": [
                  "Static ordered-estimator timing is a surrogate, not Recall-QPS.",
                  "Formal selection requires frozen full-graph artifacts and audit queries."
              ]}
    with atomic_output_dir(args.out) as partial:
        report_path = partial / "summary.json"
        _write_json(report_path, report)
        _complete(partial, "summarize", [report_path])
    print(json.dumps({"out": str(Path(args.out).resolve())}))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Unified edge-estimation pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--assets", required=True)
    pre.add_argument("--config", required=True)
    pre.add_argument("--out", required=True)
    pre.add_argument("--hash-assets", action="store_true")
    pre.add_argument("--ledger")
    pre.set_defaults(function=preflight)
    val = commands.add_parser("validate")
    val.add_argument("--events", required=True)
    val.add_argument("--labels")
    val.add_argument("--ranges")
    val.add_argument("--artifact")
    val.add_argument("--queries")
    val.add_argument("--runner")
    val.add_argument("--out", required=True)
    val.add_argument("--ledger")
    val.set_defaults(function=validate)
    timing = commands.add_parser("bench")
    timing.add_argument("--events", required=True)
    timing.add_argument("--runner", required=True)
    timing.add_argument("--repeats", type=int, default=5)
    timing.add_argument("--artifact")
    timing.add_argument("--matrix")
    timing.add_argument("--queries")
    timing.add_argument("--out", required=True)
    timing.add_argument("--ledger")
    timing.set_defaults(function=bench)
    cap = commands.add_parser("capture")
    cap.add_argument("--assets", required=True)
    cap.add_argument("--config", required=True)
    cap.add_argument("--split", required=True)
    cap.add_argument("--split-name")
    cap.add_argument("--runner", required=True)
    cap.add_argument("--dimension", type=int)
    cap.add_argument("--k", type=int)
    cap.add_argument("--ef", type=int)
    cap.add_argument("--out", required=True)
    cap.add_argument("--ledger")
    cap.set_defaults(function=capture)
    quality_command = commands.add_parser("quality")
    quality_command.add_argument("--scores")
    quality_command.add_argument("--runner")
    quality_command.add_argument("--events")
    quality_command.add_argument("--labels")
    quality_command.add_argument("--artifact")
    quality_command.add_argument("--queries")
    quality_command.add_argument("--alpha", action="append", required=True)
    quality_command.add_argument("--out", required=True)
    quality_command.add_argument("--ledger")
    quality_command.set_defaults(function=quality)
    summary = commands.add_parser("summarize")
    summary.add_argument("--quality")
    summary.add_argument("--timing")
    summary.add_argument("--out", required=True)
    summary.add_argument("--ledger")
    summary.set_defaults(function=summarize_outputs)
    catalog_command = commands.add_parser("catalog")
    catalog_command.add_argument("--assets", required=True)
    catalog_command.add_argument("--dimension", required=True, type=int)
    catalog_command.add_argument("--runner", required=True)
    catalog_command.add_argument("--out", required=True)
    catalog_command.add_argument("--ledger")
    catalog_command.set_defaults(function=catalog)
    train = commands.add_parser("train-encode")
    train.add_argument("--config", required=True); train.add_argument("--assets", required=True)
    train.add_argument("--catalog", required=True); train.add_argument("--out", required=True)
    train.add_argument("--trainer-python")
    train.add_argument("--ledger")
    train.set_defaults(function=train_encode_command)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        code = args.function(args)
        if args.ledger:
            append_ledger(args.ledger, args.command,
                          "completed" if code == 0 else "blocked",
                          {"exit_code": code, "out": str(Path(args.out).resolve())})
        return code
    except BaseException as error:
        if getattr(args, "ledger", None):
            append_ledger(args.ledger, args.command, "failed",
                          {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
