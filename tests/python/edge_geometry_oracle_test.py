"""Independent float64 geometry oracle for OPQ and CR-OPQ native artifacts.

Synthetic fixtures only: no training and no real dev/select/test queries.
The reference reconstructs vectors, never reads stored anchors/offsets as truth.
Run with the existing native runner; Windows results do not replace Linux BLAS acceptance.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.edge_estimation.contracts import EventRecord, write_events, sha256_file

ATOL, RTOL = 1e-4, 1e-5
M, K, NC = 32, 256, 256
CATALOG = bytes(range(32))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_id(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def queries_file(path, queries):
    with path.open("wb") as stream:
        for row in queries:
            stream.write(struct.pack("<I", len(row)))
            stream.write(np.asarray(row, dtype="<f4").tobytes())


def events_file(path, queries, source, order, edge_count):
    events = []
    for qid in order:
        current = float(np.sum((queries[qid].astype(np.float64) - source) ** 2))
        def add(kind, flags=0, **fields):
            events.append(EventRecord(kind, flags, 0 if kind == 3 else -1,
                                      len(events), int(qid), **fields))
        add(1)
        add(2, expansion_id=0, source_id=0, source_degree=edge_count, d_current=current)
        for eid in range(edge_count):
            add(3, 7, expansion_id=0, source_id=0, target_id=eid + 1,
                source_degree=edge_count, neighbor_slot=eid, edge_id=eid,
                d_current=current, threshold_before=10.0)
        add(5)
    write_events(path, queries.shape[1], CATALOG, events)
    return events


def publish(path, method, books, rotation, centers, codes, lengths, h, source,
            mutation=None):
    """Write the documented binary layouts without calling production encoders."""
    path.mkdir()
    d = rotation.shape[0]
    books.astype("<f4").tofile(path / "codebook.f32le")
    (rotation.T if mutation == "transpose_rotation" else rotation).astype("<f4").tofile(
        path / "rotation.f32le")
    cr = method == "ivf_opq"
    if cr:
        (np.zeros_like(centers) if mutation == "omit_centers" else centers).astype("<f4").tofile(
            path / "centers.f32le")
    records = bytearray()
    for eid, length in enumerate(lengths):
        cid = (0, 1, 127, 255, 0)[eid]
        anchor = float(source @ h[eid])
        stored = length * length + 2 * length * anchor if cr else anchor
        if mutation == "wrong_offset":
            stored += 1.0
        records += struct.pack("<ddI", length, stored, cid) if cr else struct.pack("<dd", length, stored)
        records += codes[eid].tobytes()
    (path / "edges.bin").write_bytes(records)
    files = {p.name: file_id(p) for p in sorted(path.iterdir())}
    dump(path / "manifest.json", {"synthetic_fixture": True, "files": files})
    cfg = {"backend": method, "dimension": d, "m": M, "nbits": 8,
           "edge_count": len(lengths), "coverage": "full_graph",
           "catalog_identity": CATALOG.hex()}
    if cr:
        cfg.update(format="uq-ivf-edge/1", nlist=NC, qjl_bits=0, rotation=1,
                   record_size=20 + M, manifest_sha256=sha256_file(path / "manifest.json"))
        cfg.update({name + "_sha256": value["sha256"] for name, value in files.items()})
    else:
        cfg.update(format="uq-rotated-pq/1", dsub=d // M, record_size=16 + M,
                   rotation_layout="row_major_r_times_column", rotation_bias="none",
                   codebook="codebook.f32le", rotation="rotation.f32le", records="edges.bin",
                   codebook_sha256=files["codebook.f32le"]["sha256"],
                   rotation_sha256=files["rotation.f32le"]["sha256"],
                   records_sha256=files["edges.bin"]["sha256"])
    (path / "native.cfg").write_text("".join(f"{k}={v}\n" for k, v in cfg.items()), encoding="ascii")
    (path / "complete.sha256").write_text(sha256_file(path / "native.cfg") + "\n", encoding="ascii")


def invoke(native, args, log, reject=None, expected_code=0):
    result = subprocess.run([str(native), *map(str, args)], cwd=ROOT,
                            text=True, encoding="utf-8", errors="replace", capture_output=True)
    log.write_text(result.stdout, encoding="utf-8")
    log.with_suffix(log.suffix + ".stderr.txt").write_text(result.stderr, encoding="utf-8")
    if reject is None:
        require(result.returncode == expected_code, f"{log}: exit={result.returncode}\n{result.stderr}")
    else:
        require(result.returncode != 0 and reject in result.stderr,
                f"{log}: expected rejection containing {reject!r}")
    return result.stdout


def scores(native, artifact, events_path, queries_path, events, log):
    rows = list(csv.DictReader(io.StringIO(invoke(native,
        ["scores-artifact", artifact, events_path, queries_path], log))))
    expected_ids = [e.event_id for e in events if e.kind == 3]
    require([int(row["event_id"]) for row in rows] == expected_ids,
            "Scores must cover every candidate exactly once, in order")
    return np.asarray([float(row["estimate"]) for row in rows], dtype=np.float64)


def comparison(observed, expected):
    valid = np.isfinite(expected)
    require(np.all(np.isfinite(observed[valid])), "Finite geometry produced nonfinite scores")
    require(np.all(np.isnan(observed[~valid])), "Zero edge did not return fallback sentinel")
    error = np.abs(observed[valid] - expected[valid])
    allowed = ATOL + RTOL * np.abs(expected[valid])
    return error, error / allowed


def dimension_check(out, native, d):
    out.mkdir()
    rng = np.random.default_rng(20261009 + d)
    queries = rng.normal(0, .1, (17, d)).astype("<f4")
    queries[0] = 0
    source = rng.normal(0, .1, d)
    edges = rng.normal(0, .05, (4, d))
    lengths = np.r_[np.linalg.norm(edges, axis=1), 0.0]
    books = rng.normal(0, .01, (M, K, d // M)).astype("<f4")
    centers = rng.normal(0, .02, (NC, d)).astype("<f4")
    # Dense, nonsymmetric rotation, rounded once to the stored float32 representation.
    rotation = np.linalg.qr(rng.normal(size=(d, d)))[0].astype("<f4")
    require(not np.allclose(rotation, rotation.T), "Rotation fixture must distinguish transpose")
    codes = rng.integers(0, K, (5, M), dtype=np.uint8)
    codes[:, 0], codes[:, 1] = 0, 255
    order = rng.permutation(17).tolist()
    queries_path, events_path = out / "queries.fvecs", out / "events.bin"
    queries_file(queries_path, queries)
    events = events_file(events_path, queries, source, order, 5)
    candidates = [e for e in events if e.kind == 3]
    np.savez(out / "reference.npz", source=source, edges=edges, lengths=lengths,
             queries=queries, books=books, centers=centers, rotation=rotation, codes=codes)
    result = {}
    for method in ("opq", "ivf_opq"):
        z = np.asarray([np.concatenate([books[j, code[j]].astype(np.float64)
                        for j in range(M)]) for code in codes])
        # R maps a column vector to rotated space, hence reconstruction is z @ R.
        h = z @ rotation.astype(np.float64)
        if method == "ivf_opq":
            h += centers[[0, 1, 127, 255, 0]].astype(np.float64)
        artifact = out / method
        publish(artifact, method, books, rotation, centers, codes, lengths, h, source)
        expected = np.asarray([
            np.nan if lengths[e.edge_id] == 0 else
            float(np.sum((queries[e.query_id].astype(np.float64) - source) ** 2))
            + lengths[e.edge_id] ** 2
            - 2 * lengths[e.edge_id] * np.dot(queries[e.query_id].astype(np.float64) - source,
                                              h[e.edge_id])
            for e in candidates])
        observed = scores(native, artifact, events_path, queries_path, events,
                          out / f"{method}-scores.csv")
        error, ratio = comparison(observed, expected)
        require(np.all(ratio <= 1), f"D={d} {method}: independent geometric formula mismatch")
        parity = {}
        for batch in (1, 8, 128):
            report = json.loads(invoke(native, ["validate-batch-artifact", artifact, events_path,
                queries_path, batch, ATOL, RTOL], out / f"{method}-batch{batch}.json"))
            require(report["valid"] is True and report["compared_count"] == 17 * 4
                    and report["nonfinite_mismatch_count"] == 0
                    and report["tolerance_failure_count"] == 0,
                    f"D={d} {method} batch={batch}: parity failure")
            parity[str(batch)] = report
        replay = json.loads(invoke(native, ["bench-artifact", events_path, artifact, queries_path, 1],
                                   out / f"{method}-fallback-counters.json"))
        require(replay["eligible_events"] == 85 and replay["fallback_count"] == 17,
                "Zero-edge replay fallback counts differ")
        mutations = ("transpose_rotation", "wrong_offset") + (
            ("omit_centers",) if method == "ivf_opq" else ())
        detected = {}
        for mutation in mutations:
            bad = out / f"{method}-{mutation}"
            publish(bad, method, books, rotation, centers, codes, lengths, h, source, mutation)
            observed_bad = scores(native, bad, events_path, queries_path, events,
                                  out / f"{method}-{mutation}.csv")
            _, bad_ratio = comparison(observed_bad, expected)
            require(np.any(bad_ratio > 1), f"Oracle failed to detect {method}/{mutation}")
            detected[mutation] = int(np.count_nonzero(bad_ratio > 1))

        for label, value in (("nan", np.nan), ("inf", np.inf)):
            bad_queries = queries.copy()
            bad_queries[0, 0] = value
            bad_query_path = out / f"queries-{label}.fvecs"
            queries_file(bad_query_path, bad_queries)
            invoke(native, ["scores-artifact", artifact, events_path, bad_query_path],
                   out / f"{method}-{label}-rejected.txt", reject="non-finite query value")

        # Finite inputs whose float32 LUT multiplication overflows must produce
        # invalid estimates, and the replay must count fallback for every edge.
        overflow_queries = np.full((1, d), 1e20, dtype="<f4")
        overflow_query = out / f"{method}-overflow.fvecs"
        overflow_events = out / f"{method}-overflow-events.bin"
        queries_file(overflow_query, overflow_queries)
        overflow_source = np.zeros(d)
        oe = events_file(overflow_events, overflow_queries, overflow_source, [0], 5)
        overflow_artifact = out / f"{method}-overflow"
        publish(overflow_artifact, method, np.full_like(books, 1e30), np.eye(d, dtype="<f4"),
                np.zeros_like(centers), codes, lengths, np.zeros_like(h), overflow_source)
        ov = scores(native, overflow_artifact, overflow_events, overflow_query, oe,
                    out / f"{method}-overflow-scores.csv")
        require(np.all(np.isnan(ov)), "Overflow did not return invalid-score sentinel")
        counts = json.loads(invoke(native,
            ["bench-artifact", overflow_events, overflow_artifact, overflow_query, 1],
            out / f"{method}-overflow-counters.json"))
        require(counts["eligible_events"] == 5 and counts["fallback_count"] == 5,
                "Overflow replay fallback counts differ")
        overflow_parity = json.loads(invoke(native,
            ["validate-batch-artifact", overflow_artifact, overflow_events, overflow_query, 128,
             ATOL, RTOL], out / f"{method}-overflow-parity.json", expected_code=3))
        require(overflow_parity["valid"] is False and overflow_parity["compared_count"] == 0
                and overflow_parity["nonfinite_mismatch_count"] == 0
                and overflow_parity["tolerance_failure_count"] == 0,
                "Overflow scalar/batch fallback differs")
        result[method] = {
            "finite_formula_comparisons": int(error.size), "max_absolute_error": float(error.max()),
            "max_error_over_tolerance": float(ratio.max()), "zero_edge_fallbacks": 17,
            "nonfinite_queries_rejected": ["nan", "inf"], "overflow_fallbacks": 5,
            "negative_controls_detected": detected, "batch_parity": parity,
            "overflow_batch_parity": overflow_parity,
        }
        print(f"GEOMETRY_OK D={d} method={method} compared={error.size} "
              f"max_abs={error.max():.6g}", flush=True)
    report = {"passed": True, "dimension": d, "m": M, "nbits": 8, "coarse_centers": NC,
              "query_count": 17, "query_order": order, "absolute_tolerance": ATOL,
              "relative_tolerance": RTOL, "methods": result}
    dump(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[128, 960])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    require(len(set(args.dimensions)) == len(args.dimensions)
            and all(d in (96, 128, 960) for d in args.dimensions), "Supported dimensions: 96, 128, 960")
    native = args.native.resolve(strict=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    capabilities = json.loads(invoke(native, ["capabilities"], out / "capabilities.json"))
    reports = {str(d): dimension_check(out / f"d{d}", native, d) for d in args.dimensions}
    report = {"passed": True, "synthetic_only": True, "real_queries_used": False,
              "performance_result": False, "native": file_id(native), "test_source": file_id(Path(__file__)),
              "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "capabilities": capabilities,
              "reference": "float64: ||q-c||^2 + ||edge||^2 - 2*||edge||*(q-c).u_hat",
              "scope": "native estimator geometry and replay fallback; not full HNSW top-k correctness",
              "dimensions": reports}
    dump(out / "report.json", report)
    print("INDEPENDENT_GEOMETRY_ORACLE_OK", flush=True)


if __name__ == "__main__":
    main()
