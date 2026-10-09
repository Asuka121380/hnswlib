"""Generate immutable high-dimensional data controls and v2 protocols from evidence.

No source is marked unused unless a reviewer supplies an explicit review document.
Default data controls intentionally contain uncertain review records and will not
pass materialize until a real review is supplied in a new control directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex

from scripts.datasets.prepare_highdim import verify_conversion
from scripts.edge_estimation.contracts import atomic_output_dir
from scripts.edge_estimation.final_study.common import cli, identity, load, positive, seal, write
from scripts.edge_estimation.final_study.schema import protocol

NEW_DATASETS = ("bioasq1024_cos_v1", "dbpedia_te3l1536_holdout_cos_v1")
MAIN_DATASETS = ("gist1m", "sift1m", *NEW_DATASETS)
DEFAULT_GRID = [50, 100, 200, 400, 600, 900, 1300, 1800, 2400]


def split_counts(doc):
    n = doc["n_query_candidates"]
    if doc["profile"] == "bioasq1024":
        if n < 2500:
            raise ValueError("BioASQ has fewer than 2500 eligible queries; a new research decision is required")
        return (500, 500, min(2000, n - 1000), 100)
    if n != 10000:
        raise ValueError("formal DBpedia controls require exactly 10000 held-out candidates")
    return (2000, 2000, 6000, 100)


def review_for(path, dataset, source_sha, count):
    review = load(path)
    if (review.get("dataset_id") != dataset or review.get("source_sha256") != source_sha
            or review.get("range") != [0, count] or review.get("status") != "unused_confirmed"
            or review.get("review_complete") is not True
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or not isinstance(review.get("basis"), str) or not review["basis"].strip()):
        raise ValueError("review must explicitly bind this candidate hash/range and document reviewer/basis")
    return review


def data_controls(args):
    conversions = [(Path(p).resolve(), verify_conversion(p, args.allow_synthetic)) for p in args.conversions]
    ids = [d["dataset_id"] for _, d in conversions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate conversion dataset")
    reviews = {}
    for value in args.review_evidence:
        dataset, path = value.split("=", 1)
        if dataset not in ids or dataset in reviews:
            raise ValueError("unknown/duplicate review dataset")
        reviews[dataset] = Path(path).resolve(strict=True)
    roots = [str(Path(p).resolve(strict=True)) for p in args.history_root]
    synthetic = any(d.get("synthetic_only", False) for _, d in conversions)
    if args.synthetic_counts and not all(d.get("synthetic_only") for _, d in conversions):
        raise ValueError("small split counts are restricted to synthetic inputs")
    destination = Path(args.out).resolve()
    with atomic_output_dir(destination) as root:
        registry, splits, ledger = {}, {}, []
        for path, doc in conversions:
            dataset = doc["dataset_id"]
            base = doc["outputs"]["base.fvecs"]
            query = doc["outputs"]["query_candidates.fvecs"]
            source = doc["query_source"]
            counts = tuple(args.synthetic_counts) if args.synthetic_counts else split_counts(doc)
            for value in counts:
                positive(value, "split/warmup count")
            if sum(counts[:3]) > doc["n_query_candidates"] or counts[3] >= counts[0]:
                raise ValueError("insufficient queries or invalid warmup count")
            registry[dataset] = {
                "dimension": doc["dimension"], "n_base": doc["n_base"], "metric": "squared_l2",
                "base": {"path": base["path"], "format": "fvecs"},
                "query_sources": {source: {"path": query["path"], "format": "fvecs"}},
                "preprocessing_manifest": identity(path), "synthetic_only": doc.get("synthetic_only", False),
            }
            splits[dataset] = {
                "dev": {"source": source, "count": counts[0], "warmup_count": counts[3], "allow_seen": False},
                "select": {"source": source, "count": counts[1], "allow_seen": False},
                "test": {"source": source, "count": counts[2]},
            }
            review = review_for(reviews[dataset], dataset, query["sha256"], doc["n_query_candidates"]) if dataset in reviews else None
            evidence = {"conversion_manifest": identity(path), "conversion": doc,
                        "history_reviewed": bool(review), "review": review,
                        "review_file": identity(reviews[dataset]) if review else None}
            evidence_name = f"evidence/{dataset}.json"
            seal(root / evidence_name, evidence)
            ledger.append({"dataset_id": dataset, "source": source, "source_sha256": query["sha256"],
                           "range": [0, doc["n_query_candidates"]],
                           "status": "unused_confirmed" if review else "uncertain", "review_complete": bool(review),
                           "evidence": str(destination / evidence_name)})
            if not review:
                write(root / f"review-templates/{dataset}.json", {
                    "dataset_id": dataset, "source_sha256": query["sha256"],
                    "range": [0, doc["n_query_candidates"]], "status": "uncertain",
                    "review_complete": False, "reviewer": "", "basis": "",
                    "instruction": "Complete real history review; then rerun data in a NEW directory with --review-evidence DATASET=FILE.",
                })
        write(root / "registry.highdim.json", {"datasets": registry})
        write(root / "splits.highdim.json", {"datasets": splits})
        write(root / "review-ledger.highdim.json", ledger)
        write(root / "history.highdim.json", {"roots": roots, "review_ledger": str(destination / "review-ledger.highdim.json")})
        seal(root / "controls_manifest.json", {
            "schema_version": 1, "datasets": ids, "synthetic_only": synthetic,
            "ready_for_history_audit": len(reviews) == len(ids),
            "conversions": [identity(p) for p, _ in conversions],
            "review_policy": "explicit bound reviewer document; no inferred freshness",
        })
    print(destination / "controls_manifest.json")


def protocol_controls(args):
    base = protocol(args.base_protocol)
    grids = load(args.grids) if args.grids else {key: DEFAULT_GRID[:] for key in NEW_DATASETS}
    if set(grids) != set(NEW_DATASETS):
        raise ValueError("grid file must contain exactly the two new main datasets; supplements are separate")
    for grid in grids.values():
        if (not isinstance(grid, list) or not grid
                or any(type(v) is not int or v < base["search"]["k"] for v in grid)
                or grid != sorted(set(grid))):
            raise ValueError("ef grids must be sorted unique positive integers >= k")
    if not args.pilot and not args.grids:
        raise ValueError("final protocol requires an explicit reviewed --grids file after pilot")
    if Path(args.out).exists():
        raise ValueError("protocol output exists")
    base["study_id"] = "edge-pruning-highdim-pilot-v1" if args.pilot else "edge-pruning-final-v2-highdim"
    base["selection"]["required_datasets"] = list(NEW_DATASETS if args.pilot else MAIN_DATASETS)
    # A generated protocol has no unregistered DEEP/3072 entries.
    original = base["development"]["ef_grids"]
    if not args.pilot and not all(key in original for key in ("gist1m", "sift1m")):
        raise ValueError("base protocol must declare both original datasets")
    base["development"]["ef_grids"] = {key: (grids[key] if key in grids else original[key])
                                       for key in base["selection"]["required_datasets"]}
    base["highdim_provenance"] = {"base_protocol": identity(args.base_protocol),
                                  "grids": identity(args.grids) if args.grids else None,
                                  "pilot": args.pilot, "generator": identity(__file__)}
    write(args.out, base)
    protocol(args.out)
    print(Path(args.out).resolve())


def environment_controls(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError("environment wrapper exists")
    # Shell paths are intentionally provided by the operator for the destination host.
    for value in (args.v1, args.study, args.repo):
        if not value.startswith("/") or any(c in value for c in ("\n", "\r", "\0")):
            raise ValueError("environment paths must be absolute POSIX paths")
    if args.v1 == args.study:
        raise ValueError("v2 output must not replace the v1 study")
    q = shlex.quote
    lines = ["#!/usr/bin/env bash", "# Source this wrapper; it does not submit or build anything.",
             f"source {q(args.v1 + '/control/env.sh')}",
             f"export V1={q(args.v1)}", f"export STUDY={q(args.study)}", f"export REPO={q(args.repo)}",
             'export PROTOCOL="$STUDY/control/protocol.json"',
             'export TOOLS_BUILD="$REPO/build-final-v2-tools"', 'export PERF_BUILD="$REPO/build-final-v2-perf"',
             'DATASETS=(gist1m sift1m bioasq1024_cos_v1 dbpedia_te3l1536_holdout_cos_v1)',
             'use_dataset() {', '  local ds="$1"', '  D="$STUDY/$ds"', '  case "$ds" in',
             '    gist1m|sift1m) INPUT_ROOT="$V1/$ds" ;;',
             '    bioasq1024_cos_v1|dbpedia_te3l1536_holdout_cos_v1) INPUT_ROOT="$STUDY/$ds" ;;',
             '    *) return 2 ;;', '  esac', '  DATA="$INPUT_ROOT/data/dataset_contract.json"',
             '  ASSETS="$INPUT_ROOT/assets/asset_contract.json"', '  LEDGER="$INPUT_ROOT/assets/offline_costs.json"',
             '  common=(--protocol "$PROTOCOL" --dataset "$DATA" --assets "$ASSETS"',
             '          --build "$PERF_BUILD/build_manifest.json")', '}', 'cd "$REPO"', '']
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    x = sub.add_parser("data")
    x.add_argument("--conversions", nargs="+", required=True)
    x.add_argument("--review-evidence", action="append", default=[], metavar="DATASET=REVIEW.json")
    x.add_argument("--history-root", action="append", default=[])
    x.add_argument("--allow-synthetic", action="store_true")
    x.add_argument("--synthetic-counts", nargs=4, type=int, metavar=("DEV", "SELECT", "TEST", "WARMUP"))
    x.add_argument("--out", required=True)
    x = sub.add_parser("protocol")
    x.add_argument("--base-protocol", required=True)
    x.add_argument("--grids")
    x.add_argument("--pilot", action="store_true")
    x.add_argument("--out", required=True)
    x = sub.add_parser("environment")
    for key in ("v1", "study", "repo", "out"):
        x.add_argument("--" + key, required=True)
    args = p.parse_args()
    {"data": data_controls, "protocol": protocol_controls, "environment": environment_controls}[args.stage](args)


if __name__ == "__main__":
    cli(main)
