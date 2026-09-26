from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.edge_estimation.contracts import (
        append_ledger, atomic_output_dir, file_entry, load_strict_json, read_events,
        read_labels, read_query_ranges, sha256_file, validate_config_v1, validate_dataset,
    )
    from scripts.edge_estimation.capabilities import environment_capabilities
else:
    from .contracts import (
        append_ledger, atomic_output_dir, file_entry, load_strict_json, read_events,
        read_labels, read_query_ranges, sha256_file, validate_config_v1, validate_dataset,
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


STAGE_ASSETS = {
    "generic": ("index", "queries"),
    "catalog": ("index",),
    "capture": ("index", "queries"),
    "train": ("base", "internal_to_label"),
    "quality": ("queries",),
    "formal": ("index", "queries", "base", "internal_to_label", "ground_truth"),
}


def _asset_path_and_expected(name: str, value: Any,
                             expected: Mapping[str, Any]) -> tuple[Path, str | None]:
    expected_hash = expected.get(name)
    if isinstance(value, str):
        raw_path = value
    elif isinstance(value, Mapping):
        raw_path = value.get("path")
        expected_hash = value.get("sha256", expected_hash)
    else:
        raise ValueError(f"asset {name} must be a path string or path/sha256 object")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"asset {name} has no path")
    if expected_hash is not None and (not isinstance(expected_hash, str) or
                                      len(expected_hash) != 64):
        raise ValueError(f"asset {name} expected sha256 must be 64 hex characters")
    return Path(raw_path).expanduser().resolve(), expected_hash


def preflight(args: argparse.Namespace) -> int:
    assets = load_strict_json(args.assets)
    config = load_strict_json(args.config)
    validate_config_v1(config)
    _require_keys(assets, ["schema_version"], "assets")
    if assets["schema_version"] != 1 or config["schema_version"] != 1:
        raise ValueError("only schema_version=1 is supported")
    required_assets = STAGE_ASSETS[args.stage]
    missing_keys = [key for key in required_assets if key not in assets]
    expected = assets.get("expected_sha256", {})
    if not isinstance(expected, Mapping):
        raise ValueError("assets.expected_sha256 must be an object")
    asset_status: dict[str, object] = {}
    for key, raw_path in assets.items():
        if key in ("schema_version", "expected_sha256"):
            continue
        path, expected_hash = _asset_path_and_expected(key, raw_path, expected)
        asset_status[key] = {"path": str(path), "exists": path.is_file(),
                             "size": path.stat().st_size if path.is_file() else None}
        if expected_hash is not None:
            asset_status[key]["expected_sha256"] = expected_hash.lower()
        if (args.hash_assets or expected_hash is not None) and path.is_file():
            actual_hash = sha256_file(path)
            asset_status[key]["sha256"] = actual_hash
            asset_status[key]["identity_match"] = (
                expected_hash is None or actual_hash == expected_hash.lower())
    missing = missing_keys + [key for key in required_assets
                              if key in asset_status and not asset_status[key]["exists"]]
    mismatched = [key for key, value in asset_status.items()
                  if value.get("identity_match") is False]
    unfrozen = ([key for key in required_assets
                 if key in asset_status and "expected_sha256" not in asset_status[key]]
                if args.stage == "formal" else [])
    if missing:
        status = "blocked_missing_assets"
    elif mismatched:
        status = "blocked_identity_mismatch"
    elif unfrozen:
        status = "blocked_missing_expected_identity"
    else:
        status = "ready"
    report = {
        "schema_version": 1,
        "stage": "preflight",
        "requested_stage": args.stage,
        "required_assets": list(required_assets),
        "status": status,
        "missing_assets": missing,
        "identity_mismatches": mismatched,
        "missing_expected_identities": unfrozen,
        "formal_asset_identity_complete": (
            args.stage == "formal" and not missing and not mismatched and not unfrozen and
            all(asset_status[key].get("identity_match") is True
                for key in required_assets)),
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
    return 0 if status == "ready" else 3


def convert_split(args: argparse.Namespace) -> int:
    source = load_strict_json(args.input)
    raw_splits = source.get("splits", source)
    if not isinstance(raw_splits, Mapping):
        raise ValueError("split input must be an object or contain a splits object")
    names = ("development", "selection", "audit")
    converted: dict[str, list[int]] = {}
    seen: set[int] = set()
    for name in names:
        values = raw_splits.get(name)
        if (not isinstance(values, list) or not values or
                not all(isinstance(value, int) and not isinstance(value, bool)
                        for value in values)):
            raise ValueError(f"split {name} must be a non-empty integer list")
        if len(set(values)) != len(values):
            raise ValueError(f"split {name} contains duplicate query IDs")
        overlap = seen.intersection(values)
        if overlap:
            raise ValueError(f"split {name} overlaps an earlier split")
        if any(value < 0 or value >= args.query_count for value in values):
            raise ValueError(f"split {name} contains an out-of-range query ID")
        converted[name] = list(values)
        seen.update(values)
    expected = set(range(args.query_count))
    if seen != expected:
        raise ValueError("split groups must cover every query ID exactly once")
    result = {
        "schema_version": 1,
        "source": {"path": str(Path(args.input).resolve()),
                   "sha256": sha256_file(args.input)},
        "query_count": args.query_count,
        "split_counts": {name: len(converted[name]) for name in names},
        "splits": converted,
    }
    with atomic_output_dir(args.out) as partial:
        result_path = partial / "split.json"
        _write_json(result_path, result)
        _complete(partial, "convert-split", [result_path])
    print(json.dumps({"query_count": args.query_count,
                      "split_counts": result["split_counts"]}))
    return 0


def wrap_legacy(args: argparse.Namespace) -> int:
    catalog = load_strict_json(Path(args.catalog) / "manifest.json")
    _require_keys(catalog, ["identity_sha256", "edge_count"],
                  "catalog manifest")
    sidecar_source = Path(args.sidecar).resolve()
    residual_source = Path(args.residual).resolve() if args.residual else None
    if not sidecar_source.is_file() or (residual_source and not residual_source.is_file()):
        raise FileNotFoundError("legacy sidecar or residual companion is missing")
    backend = "pq_qjl_legacy" if residual_source else "pq_legacy"
    artifact_format = ("uq-pq-qjl-legacy/1" if residual_source else
                       "uq-pq-legacy/1")
    with atomic_output_dir(args.out) as partial:
        payload = partial / "legacy"
        payload.mkdir()
        sidecar = payload / "sidecar.bin"
        shutil.copyfile(sidecar_source, sidecar)
        lines = [
            f"format={artifact_format}", f"backend={backend}",
            "coverage=full_graph", f"edge_count={int(catalog['edge_count'])}",
            f"catalog_identity={catalog['identity_sha256']}",
            "sidecar=legacy/sidecar.bin", f"sidecar_sha256={sha256_file(sidecar)}",
        ]
        files = {"sidecar": _artifact_file_entry(sidecar, "legacy/sidecar.bin")}
        if residual_source:
            residual = payload / "residual.bin"
            shutil.copyfile(residual_source, residual)
            lines.extend(["residual=legacy/residual.bin",
                          f"residual_sha256={sha256_file(residual)}"])
            files["residual"] = _artifact_file_entry(residual, "legacy/residual.bin")
        native = partial / "native.cfg"
        native.write_text("\n".join(lines) + "\n", encoding="ascii")
        files["native_config"] = _artifact_file_entry(native, "native.cfg")
        manifest = partial / "manifest.json"
        _write_json(manifest, {
            "schema_version": 1, "format_family": artifact_format.rsplit("/", 1)[0],
            "format_version": 1, "algorithm": backend, "provider": "legacy_v0",
            "edge_catalog_identity": catalog["identity_sha256"],
            "edge_count": int(catalog["edge_count"]), "coverage": "full_graph",
            "source_identities": {
                "sidecar": {"path": str(sidecar_source),
                            "sha256": sha256_file(sidecar_source)},
                **({"residual": {"path": str(residual_source),
                                  "sha256": sha256_file(residual_source)}}
                   if residual_source else {}),
            },
            "files": files,
        })
        files["manifest"] = _artifact_file_entry(manifest, "manifest.json")
        _complete(partial, "wrap-legacy", [sidecar, native, manifest] +
                  ([partial / "legacy" / "residual.bin"] if residual_source else []))
    print(json.dumps({"backend": backend, "out": str(Path(args.out).resolve())}))
    return 0


def _artifact_file_entry(path: Path, relative: str) -> dict[str, Any]:
    return {**file_entry(path), "path": relative}


def validate(args: argparse.Namespace) -> int:
    if args.artifact:
        if not args.runner or not args.events or not args.queries:
            raise ValueError("artifact validation requires --runner, --events and --queries")
        command = [str(Path(args.runner).resolve()), "validate-artifact",
                   str(Path(args.artifact).resolve()), str(Path(args.events).resolve()),
                   str(Path(args.queries).resolve())]
        report = json.loads(subprocess.run(command, check=True, text=True,
                                           capture_output=True).stdout)
        report["input_identities"] = {
            "events_sha256": sha256_file(args.events),
            "queries_sha256": sha256_file(args.queries),
            "artifact_manifest_sha256": sha256_file(Path(args.artifact) / "manifest.json"),
            "artifact_native_cfg_sha256": sha256_file(Path(args.artifact) / "native.cfg"),
        }
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
              "query_count": len(ranges), "dimension": event_header.dimension,
              "input_identities": {
                  "events_sha256": sha256_file(args.events),
                  "labels_sha256": sha256_file(args.labels),
                  "ranges_sha256": sha256_file(args.ranges),
              }}
    with atomic_output_dir(args.out) as partial:
        report_path = partial / "validation.json"
        _write_json(report_path, report)
        _complete(partial, "validate", [report_path])
    print(json.dumps(report))
    return 0


def _timing_evidence(validation_path: str | None, quality_path: str | None,
                     events: str, queries: str | None, artifact: str | None,
                     formal: bool) -> dict[str, Any]:
    validation_evidence = quality_evidence = None
    expected = {"events_sha256": sha256_file(events)}
    if queries:
        expected["queries_sha256"] = sha256_file(queries)
    if artifact:
        expected["artifact_manifest_sha256"] = sha256_file(
            Path(artifact) / "manifest.json")
        expected["artifact_native_cfg_sha256"] = sha256_file(
            Path(artifact) / "native.cfg")
    if validation_path:
        validation_value = load_strict_json(validation_path)
        if validation_value.get("valid") is not True:
            raise ValueError("validation evidence is not a passing artifact validation")
        identities = validation_value.get("input_identities", {})
        if any(identities.get(key) != value for key, value in expected.items()):
            raise ValueError("validation evidence identity does not match timing inputs")
        validation_evidence = {"path": str(Path(validation_path).resolve()),
                               "sha256": sha256_file(validation_path)}
    if quality_path:
        quality_value = load_strict_json(quality_path)
        if not isinstance(quality_value.get("summaries"), list) or not quality_value["summaries"]:
            raise ValueError("quality evidence has no summaries")
        identities = quality_value.get("input_identities", {})
        if any(identities.get(key) != value for key, value in expected.items()):
            raise ValueError("quality evidence identity does not match timing inputs")
        quality_evidence = {"path": str(Path(quality_path).resolve()),
                            "sha256": sha256_file(quality_path)}
    if formal and (validation_evidence is None or quality_evidence is None):
        raise ValueError("formal timing requires --validation and --quality-report")
    return {"validation": validation_evidence, "quality": quality_evidence,
            "formal_admitted": bool(formal), "verified_inputs": expected}


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
        method_evidence = {
            str(item["name"]): _timing_evidence(
                item.get("validation"), item.get("quality_report"), args.events,
                args.queries, str(Path(item["artifact"]).resolve()), args.formal)
            for item in methods
        }
        blocks = int(matrix.get("blocks", 5))
        repeats = int(matrix.get("repeats", 5))
        seed = int(matrix.get("seed", 20260924))
        reference = str(matrix.get("reference", names[0]))
        dataset_id = sha256_file(args.events)
        build_id = sha256_file(args.runner)
        records = []
        for slot in paired_schedule(names, blocks, repeats, seed):
            command = [str(Path(args.runner).resolve()), "bench-artifact",
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
                            "memory_report": value["memory_report"],
                            "evidence": method_evidence[str(slot["method"])]})
        result = {"schema_version": 1, "mode": "randomized_paired_blocks",
                  "raw_records": records,
                  "summary": summarize_paired(records, reference),
                  "evidence": {"by_method": method_evidence,
                               "formal_admitted": bool(args.formal)},
                  "input_identities": {"events_sha256": dataset_id,
                                       "runner_sha256": build_id,
                                       "queries_sha256": sha256_file(args.queries)}}
        with atomic_output_dir(args.out) as partial:
            result_path = partial / "result.json"
            _write_json(result_path, result)
            _complete(partial, "bench", [result_path])
        print(json.dumps({"record_count": len(records), "reference": reference}))
        return 0
    if args.artifact:
        if not args.queries:
            raise ValueError("artifact benchmark requires --queries")
        command = [str(Path(args.runner).resolve()), "bench-artifact",
                   str(Path(args.events).resolve()), str(Path(args.artifact).resolve()),
                   str(Path(args.queries).resolve()), str(args.repeats)]
    else:
        command = [str(Path(args.runner).resolve()), "bench",
                   str(Path(args.events).resolve()), str(args.repeats)]
    completed = subprocess.run(command, check=True, text=True,
                               capture_output=True)
    result = json.loads(completed.stdout)
    evidence = _timing_evidence(
        args.validation, args.quality_report, args.events, args.queries,
        args.artifact, args.formal)
    result["evidence"] = evidence
    result["quality_valid"] = evidence["quality"] is not None
    result["formal_validation_passed"] = evidence["formal_admitted"]
    result["input_identities"] = {
        "events_sha256": sha256_file(args.events),
        "runner_sha256": sha256_file(args.runner),
        **({"queries_sha256": sha256_file(args.queries)} if args.queries else {}),
    }
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
            command = [str(Path(args.runner).resolve()), "quality",
                       str(Path(args.events).resolve()), str(Path(args.labels).resolve()),
                       str(Path(args.artifact).resolve()), str(Path(args.queries).resolve()), str(alpha)]
            results.append(json.loads(subprocess.run(command, check=True, text=True,
                                                     capture_output=True).stdout))
        report = {"schema_version": 1, "native": True, "summaries": results,
                  "input_identities": {
                      "events_sha256": sha256_file(args.events),
                      "labels_sha256": sha256_file(args.labels),
                      "queries_sha256": sha256_file(args.queries),
                      "artifact_manifest_sha256": sha256_file(
                          Path(args.artifact) / "manifest.json"),
                      "artifact_native_cfg_sha256": sha256_file(
                          Path(args.artifact) / "native.cfg"),
                  }}
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
    pre.add_argument("--stage", choices=tuple(STAGE_ASSETS), default="generic")
    pre.add_argument("--ledger")
    pre.set_defaults(function=preflight)
    split_command = commands.add_parser("convert-split")
    split_command.add_argument("--input", required=True)
    split_command.add_argument("--query-count", type=int, required=True)
    split_command.add_argument("--out", required=True)
    split_command.add_argument("--ledger")
    split_command.set_defaults(function=convert_split)
    legacy = commands.add_parser("wrap-legacy")
    legacy.add_argument("--sidecar", required=True)
    legacy.add_argument("--residual")
    legacy.add_argument("--catalog", required=True)
    legacy.add_argument("--out", required=True)
    legacy.add_argument("--ledger")
    legacy.set_defaults(function=wrap_legacy)
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
    timing.add_argument("--validation")
    timing.add_argument("--quality-report")
    timing.add_argument("--formal", action="store_true")
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
