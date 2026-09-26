from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.edge_estimation.contracts import (  # type: ignore
        atomic_output_dir, file_entry, load_strict_json)
    from scripts.edge_estimation.quality_policy import (  # type: ignore
        analyze_native_summary, select_operating_points, sweep_alphas,
        validate_quality_policy)
else:
    from .contracts import atomic_output_dir, file_entry, load_strict_json
    from .quality_policy import (analyze_native_summary,
                                 select_operating_points, sweep_alphas,
                                 validate_quality_policy)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def run_quality_sweep(*, runner: Path, events: Path, labels: Path,
                      queries: Path, artifact: Path, policy_path: Path,
                      output: Path) -> None:
    policy = load_strict_json(policy_path)
    validate_quality_policy(policy)
    required_files = {
        "runner": runner,
        "events": events,
        "labels": labels,
        "queries": queries,
        "artifact_manifest": artifact / "manifest.json",
        "artifact_native_config": artifact / "native.cfg",
        "policy": policy_path,
    }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing quality sweep inputs: " + ", ".join(missing))
    summaries: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    backend: str | None = None
    for alpha in sweep_alphas(policy):
        command = [str(runner.resolve()), "quality", str(events.resolve()),
                   str(labels.resolve()), str(artifact.resolve()),
                   str(queries.resolve()), format(alpha, ".12g")]
        completed = subprocess.run(command, check=True, text=True,
                                   capture_output=True)
        summary = json.loads(completed.stdout)
        if summary.get("schema_version") != 1:
            raise ValueError("native quality output has the wrong schema version")
        if abs(float(summary.get("alpha", -1.0)) - alpha) > 1e-12:
            raise ValueError("native quality output alpha does not match request")
        current_backend = str(summary.get("backend", ""))
        if not current_backend or (backend is not None and current_backend != backend):
            raise ValueError("native quality backend identity changed during sweep")
        backend = current_backend
        summaries.append(summary)
        analyses.append(analyze_native_summary(summary))
    selection = select_operating_points(analyses, policy)
    inputs = {name: file_entry(path) for name, path in required_files.items()}
    with atomic_output_dir(output) as partial:
        quality_path = partial / "quality.json"
        selection_path = partial / "selection.json"
        policy_copy = partial / "policy.json"
        _write_json(quality_path, {
            "schema_version": 1,
            "stage": "quality-sweep",
            "protocol_id": policy["protocol_id"],
            "native": True,
            "backend": backend,
            "alphas": sweep_alphas(policy),
            "inputs": inputs,
            "summaries": summaries,
            "analyses": analyses,
        })
        _write_json(selection_path, selection)
        _write_json(policy_copy, policy)
        complete_path = partial / "complete.json"
        _write_json(complete_path, {
            "schema_version": 1,
            "stage": "quality-sweep",
            "protocol_id": policy["protocol_id"],
            "inputs": inputs,
            "outputs": {
                "quality": file_entry(quality_path),
                "selection": file_entry(selection_path),
                "policy": file_entry(policy_copy),
            },
        })
    print(json.dumps({
        "status": "complete", "backend": backend,
        "alpha_count": len(summaries), "out": str(output.resolve()),
        "selections": selection["selections"],
    }, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and select a frozen native quality-policy sweep")
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    run_quality_sweep(runner=args.runner, events=args.events,
                      labels=args.labels, queries=args.queries,
                      artifact=args.artifact, policy_path=args.policy,
                      output=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
