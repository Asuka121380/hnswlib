"""Pinned Parquet acquisition and bounded-memory, auditable high-dimensional inputs.

This module needs PyArrow only for probe/lock/convert. It never runs ANN methods,
declares queries historically unused, modifies a study protocol, or submits jobs.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import sqlite3
import struct
import sys
import time
import urllib.request

import numpy as np

from scripts.edge_estimation.contracts import atomic_output_dir
from scripts.edge_estimation.final_study.common import (
    cli, digest, identity, load, positive, seal, unseal, verify_file, write,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs/edge_estimation/final_study/highdim-sources.json"
PROFILES = ("bioasq1024", "dbpedia1536", "dbpedia3072")
NORMALIZATION = "l2-f64-to-f32"


def arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("PyArrow is required in the separate data-preparation environment") from error
    return pa, pq


def profile(name):
    spec = dict(load(CATALOG)["profiles"][name])
    spec["profile"] = name
    if "shards" in spec:
        spec["files"] = [
            {"relative_path": f"data/train-{i:05d}-of-{spec['shards']:05d}.parquet", "role": "corpus"}
            for i in range(spec["shards"])
        ]
    return spec


def source_url(spec, relative):
    if "base_url" in spec:
        return spec["base_url"] + relative
    return (f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/"
            f"{spec['revision']}/{relative}")


def metadata(path):
    _, pq = arrow()
    f = pq.ParquetFile(path)
    return {"rows": f.metadata.num_rows, "row_groups": f.metadata.num_row_groups,
            "schema": str(f.schema_arrow)}


def small_request(url, range_header, cap):
    req = urllib.request.Request(url, headers={"Range": range_header})
    with urllib.request.urlopen(req, timeout=60) as response:
        if response.status != 206:
            raise ValueError("server did not honor HTTP Range; refusing full download in probe")
        raw = response.read(cap + 1)
        if len(raw) > cap:
            raise ValueError("metadata response exceeds probe budget")
        headers = {k.lower(): v for k, v in response.headers.items()}
    return raw, headers


def remote_metadata(url):
    tail, headers = small_request(url, "bytes=-8", 8)
    if len(tail) != 8 or tail[4:] != b"PAR1":
        raise ValueError("invalid Parquet footer marker")
    length = struct.unpack("<I", tail[:4])[0]
    if not 0 < length <= 4 * 1024 * 1024:
        raise ValueError("footer outside 4 MiB metadata budget")
    raw, _ = small_request(url, f"bytes=-{length + 8}", length + 8)
    if len(raw) != length + 8 or raw[-8:] != tail:
        raise ValueError("inconsistent footer range")
    return {**metadata(io.BytesIO(b"PAR1" + raw)), "headers": headers,
            "footer_sha256": hashlib.sha256(raw).hexdigest()}


def probe(args):
    if Path(args.out).exists():
        raise ValueError("probe output exists")
    spec = profile(args.profile)
    records = []
    for entry in spec["files"]:
        url = source_url(spec, entry["relative_path"])
        info = metadata(Path(args.raw_dir) / entry["relative_path"]) if args.raw_dir else remote_metadata(url)
        records.append({**entry, "url": url, **info})
    seal(args.out, {"schema_version": 1, "profile": spec["profile"], "files": records,
                    "scope": "Parquet metadata only; vector values and full hashes not verified"})


def make_lock(spec, raw_dir, out, published_dir=None):
    raw_dir = Path(raw_dir).resolve()
    published = Path(published_dir).resolve() if published_dir else raw_dir
    entries = []
    for entry in spec["files"]:
        p = raw_dir / entry["relative_path"]
        item = {**entry, **identity(p), "url": source_url(spec, entry["relative_path"]), **metadata(p)}
        item["path"] = str(published / entry["relative_path"])
        entries.append(item)
    seal(out, {"schema_version": 1, "profile": spec["profile"],
               "dataset_id": spec["dataset_id"], "dimension": spec["dimension"],
               "vector_column": spec["vector_column"], "id_column": spec["id_column"],
               "revision": spec.get("revision"), "raw_files": entries,
               "synthetic_only": False, "acquired_at": datetime.now(timezone.utc).isoformat(),
               "source_catalog": identity(CATALOG)})


def download(args):
    spec = profile(args.profile)
    destination = Path(args.out).resolve()
    with atomic_output_dir(destination) as root:
        for entry in spec["files"]:
            target = root / entry["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            url = source_url(spec, entry["relative_path"])
            print(f"Downloading {entry['relative_path']}", flush=True)
            partial = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("xb") as output:
                copied = 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    copied += len(chunk)
                length = response.headers.get("Content-Length")
                if length is not None and copied != int(length):
                    raise ValueError("download length mismatch")
            partial.rename(target)
        make_lock(spec, root, root / "source-lock.json", destination)
    print(destination / "source-lock.json")


def lock_local(args):
    if Path(args.out).exists():
        raise ValueError("source lock exists; use a new output")
    make_lock(profile(args.profile), args.raw_dir, args.out)


def load_lock(path, expected_profile, allow_synthetic=False):
    doc = unseal(path)
    spec = profile(expected_profile)
    if doc.get("schema_version") != 1:
        raise ValueError("unsupported source lock")
    for key in ("profile", "dataset_id", "dimension", "id_column", "vector_column"):
        if doc.get(key) != spec[key]:
            raise ValueError(f"source lock/profile mismatch: {key}")
    if doc.get("revision") != spec.get("revision"):
        raise ValueError("source lock must use the pinned revision")
    synthetic = doc.get("synthetic_only", False)
    if type(synthetic) is not bool or synthetic and not allow_synthetic:
        raise ValueError("synthetic source requires --allow-synthetic and cannot be formal evidence")
    entries = doc.get("raw_files", [])
    names = [e["relative_path"] for e in entries]
    if not names or len(set(names)) != len(names):
        raise ValueError("missing/duplicate source files")
    for n in names:
        p = PurePosixPath(n)
        if p.is_absolute() or ".." in p.parts or "\\" in n or ":" in n:
            raise ValueError("invalid source relative_path")
    expected = {(e["relative_path"], e["role"]) for e in spec["files"]}
    actual = {(e["relative_path"], e["role"]) for e in entries}
    if not synthetic and actual != expected:
        raise ValueError("source lock is missing or substitutes published shards/roles")
    roles = {e["role"] for e in entries}
    if spec["kind"] == "official_queries":
        if not {"base", "query"} <= roles or not roles <= {"base", "query", "gt"}:
            raise ValueError("BioASQ requires separate base and query roles")
    elif roles != {"corpus"}:
        raise ValueError("DBpedia requires corpus shards")
    if len({str(Path(e["path"]).resolve()) for e in entries}) != len(entries):
        raise ValueError("multiple source entries refer to the same file")
    for e in entries:
        verify_file(e)
    return doc, spec


def json_line(stream, doc):
    stream.write(json.dumps(doc, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def text_hash(title, text):
    if not isinstance(title, str) or not isinstance(text, str):
        raise ValueError("DBpedia title/text must be non-null strings")
    # Equal document bodies belong to one group even under different titles.
    return digest(text)


def normalized(vector, dimension):
    if not vector.is_valid:
        return None, "null_vector"
    if len(vector.values) != dimension:
        raise ValueError(f"vector dimension differs from {dimension}")
    if vector.values.null_count:
        return None, "null_element"
    x = np.asarray(vector.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    if not np.isfinite(x).all():
        return None, "nonfinite_vector"
    # Scale first to avoid overflow/underflow for finite but extreme input values.
    scale = float(np.max(np.abs(x)))
    if scale == 0:
        return None, "zero_norm"
    # Use the specified direct float64 norm whenever representable.
    with np.errstate(over="ignore", under="ignore"):
        norm = float(np.linalg.norm(x))
    y = x / norm if np.isfinite(norm) and norm > 0 else (x / scale) / np.linalg.norm(x / scale)
    y = y.astype("<f4")
    y[y == 0] = 0
    if not np.isfinite(y).all() or not np.any(y):
        raise ValueError("normalization failed")
    return y, None


def rows(entry, spec, batch_size):
    pa, pq = arrow()
    f = pq.ParquetFile(entry["path"])
    columns = [spec["id_column"], spec["vector_column"]]
    if spec["kind"] == "heldout_documents":
        columns += ["title", "text"]
    if not set(columns) <= set(f.schema_arrow.names):
        raise ValueError(f"missing source columns in {entry['path']}")
    dtype = f.schema_arrow.field(spec["vector_column"]).type
    if not (pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype)):
        raise ValueError("vector column must be an Arrow list")
    if not pa.types.is_floating(dtype.value_type):
        raise ValueError("vector elements must be float32/float64")
    row = 0
    for batch in f.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
        for i in range(batch.num_rows):
            source_id = batch.column(0)[i].as_py()
            if spec["kind"] == "official_queries":
                valid_id = type(source_id) is int
            else:
                valid_id = isinstance(source_id, str) and bool(source_id)
            if not valid_id:
                raise ValueError("null or invalid source ID")
            origin = {"source_file": entry["relative_path"], "source_row": row,
                      "source_id": source_id, "source_role": entry["role"]}
            if len(columns) == 4:
                origin["text_sha256"] = text_hash(batch.column(2)[i].as_py(), batch.column(3)[i].as_py())
                origin["title_sha256"] = digest(batch.column(2)[i].as_py())
            try:
                vector, reason = normalized(batch.column(1)[i], spec["dimension"])
            except ValueError as error:
                raise ValueError(f"{error}; source={json.dumps(origin, ensure_ascii=False)}") from error
            yield origin, vector, reason
            row += 1
    if row != f.metadata.num_rows:
        raise ValueError("Parquet metadata/iterated row count mismatch")


class Groups:
    """Disk-backed identity/hash indexes and a compact transitive union-find."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE seen(kind TEXT, key TEXT, row_id INTEGER, signature TEXT, PRIMARY KEY(kind,key))")
        self.parent = array("Q")

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def lookup(self, kind, key, row, signature="", conflict=False):
        found = self.db.execute("SELECT row_id, signature FROM seen WHERE kind=? AND key=?", (kind,key)).fetchone()
        if found:
            if conflict and found[1] != signature:
                raise ValueError(f"conflicting duplicate source ID: {key}")
            return found[0]
        self.db.execute("INSERT INTO seen VALUES(?,?,?,?)", (kind,key,row,signature))
        return None

    def group(self, origin, vector_sha, i):
        self.parent.append(i)
        keys = [("id", json.dumps(origin["source_id"]), digest([origin["title_sha256"], origin["text_sha256"], vector_sha]), True),
                ("text", origin["text_sha256"], "", False), ("vector", vector_sha, "", False)]
        for kind, key, signature, conflict in keys:
            other = self.lookup(kind, key, i, signature, conflict)
            if other is not None:
                a, b = self.find(i), self.find(other)
                self.parent[max(a,b)] = min(a,b)


def convert(args):
    positive(args.batch_size, "batch_size")
    if args.holdout_seed < 0:
        raise ValueError("holdout seed must be nonnegative")
    started = time.perf_counter()
    source, spec = load_lock(args.source_lock, args.profile, args.allow_synthetic)
    lock_identity = identity(args.source_lock)
    if spec["kind"] == "official_queries" and args.holdout_count is not None:
        raise ValueError("BioASQ uses official queries; no implicit base holdout")
    holdout = args.holdout_count if args.holdout_count is not None else 10000
    positive(holdout, "holdout_count")
    d = spec["dimension"]
    destination = Path(args.out).resolve()
    counts = Counter()
    source_counts = Counter()
    header = struct.pack("<i", d)
    with atomic_output_dir(destination) as root, ExitStack() as stack:
        groups = Groups(root / "dedup.sqlite")
        stack.callback(groups.db.close)
        exclusions = stack.enter_context((root / "exclusions.jsonl").open("w", encoding="utf-8", newline="\n"))
        def exclude(origin, reason, **extra):
            counts[reason] += 1
            json_line(exclusions, {**origin, "reason":reason, **extra})
        streams = {}
        for role in ("base", "query", "staging"):
            streams[role] = (
                stack.enter_context((root / f"{role}.fvecs").open("wb")),
                stack.enter_context((root / f"{role}.jsonl").open("w", encoding="utf-8", newline="\n")),
            )
        n = Counter()
        def emit(role, vector, origin):
            streams[role][0].write(header + vector.tobytes())
            json_line(streams[role][1], {**origin, "row": n[role]})
            n[role] += 1
        entries = sorted(source["raw_files"], key=lambda e: (e["role"] != "base", e["relative_path"]))
        for entry in entries:
            if entry["role"] == "gt":
                continue  # Publisher GT is provenance only; project GT is recomputed.
            for origin, vector, reason in rows(entry, spec, args.batch_size):
                source_counts[entry["role"]] += 1
                if reason:
                    exclude(origin, reason)
                    continue
                sha = hashlib.sha256(vector.tobytes()).hexdigest()
                origin["vector_sha256"] = sha
                if spec["kind"] == "heldout_documents":
                    groups.group(origin, sha, n["staging"])
                    emit("staging", vector, origin)
                else:
                    role = entry["role"]
                    groups.lookup(role + "_id", json.dumps(origin["source_id"]), n[role], sha, True)
                    if role == "query":
                        match = groups.db.execute("SELECT row_id FROM seen WHERE kind='base_vector' AND key=?", (sha,)).fetchone()
                        if match:
                            exclude(origin, "base_overlap", base_row=match[0])
                            continue
                    match = groups.lookup(role + "_vector", sha, n[role])
                    if role == "query" and match is not None:
                        exclude(origin, "duplicate_query", representative_row=match)
                        continue
                    emit(role, vector, origin)
                if sum(source_counts.values()) % args.batch_size == 0:
                    groups.db.commit()
            groups.db.commit()
        for pair in streams.values():
            for stream in pair:
                stream.flush()
        if spec["kind"] == "heldout_documents":
            representatives = np.fromiter((i for i in range(n["staging"]) if groups.find(i) == i), dtype=np.int64)
            if len(representatives) <= holdout:
                raise ValueError("not enough unique corpus records to leave a nonempty base")
            # Membership is random; output order is canonical and independent of batch size.
            chosen = set(map(int, np.random.Generator(np.random.PCG64(args.holdout_seed)).choice(
                representatives, holdout, replace=False)))
            with (root / "staging.fvecs").open("rb") as vectors, (root / "staging.jsonl").open(encoding="utf-8") as origins:
                for i, line in enumerate(origins):
                    record = vectors.read(4 * (d + 1))
                    if len(record) != 4 * (d + 1):
                        raise ValueError("truncated staging vector")
                    origin = json.loads(line)
                    origin.pop("row")
                    representative = groups.find(i)
                    if representative != i:
                        exclude(origin, "duplicate_group", representative_source_ordinal=representative)
                        continue
                    vector = np.frombuffer(record, dtype="<f4", offset=4)
                    emit("query" if i in chosen else "base", vector, origin)
        if not n["base"] or not n["query"]:
            raise ValueError("conversion left an empty base/query pool")
        # Close handles before rename/publication on Windows.
        stack.close()
        (root / "base.jsonl").rename(root / "base_row_to_source_id.jsonl")
        (root / "query.jsonl").rename(root / "query_row_to_source_id.jsonl")
        (root / "query.fvecs").rename(root / "query_candidates.fvecs")
        for temporary in ("staging.fvecs", "staging.jsonl", "dedup.sqlite"):
            (root / temporary).unlink()
        # Detect input mutation during conversion before publishing successful output.
        verify_file(lock_identity)
        for entry in source["raw_files"]:
            verify_file(entry)
        outputs = {}
        for p in sorted(root.iterdir()):
            entry = identity(p)
            entry["path"] = str(destination / p.name)
            outputs[p.name] = entry
        pa, _ = arrow()
        seal(root / "conversion_manifest.json", {
            "schema_version": 1, "profile": args.profile, "dataset_id": spec["dataset_id"],
            "dimension": d, "n_base": n["base"], "n_query_candidates": n["query"],
            "metric": "squared_l2", "source_semantics": "cosine",
            "normalization": NORMALIZATION, "zero_sign": "positive", "norm_overflow_fallback": "scaled_float64",
            "source_lock": lock_identity, "raw_files": source["raw_files"], "outputs": outputs,
            "source_rows": dict(source_counts), "exclusions": dict(counts),
            "holdout_count": holdout if spec["kind"] == "heldout_documents" else None,
            "holdout_seed": args.holdout_seed if spec["kind"] == "heldout_documents" else None,
            "holdout_rng": "numpy.PCG64; canonical representative order",
            "query_source": "official_clean" if spec["kind"] == "official_queries" else "heldout",
            "text_deduplication": "exact text column, independent of title" if spec["kind"] == "heldout_documents" else None,
            "synthetic_only": source.get("synthetic_only", False), "history_reviewed": False,
            "base_query_overlap_checked": True,
            "script": identity(__file__), "source_catalog": identity(CATALOG),
            "environment": {"python": sys.version, "numpy": np.__version__, "pyarrow": pa.__version__, "platform": platform.platform()},
            "batch_size": args.batch_size, "wall_seconds": time.perf_counter() - started,
        })
    print(destination / "conversion_manifest.json")


def verify_conversion(path, allow_synthetic=False):
    doc = unseal(path)
    spec = profile(doc["profile"])
    if doc.get("schema_version") != 1 or doc.get("dataset_id") != spec["dataset_id"] or doc.get("dimension") != spec["dimension"]:
        raise ValueError("conversion manifest/profile mismatch")
    if doc.get("synthetic_only") and not allow_synthetic:
        raise ValueError("synthetic conversion cannot generate formal controls")
    for entry in doc["outputs"].values():
        verify_file(entry)
    verify_file(doc["source_lock"])
    if not doc.get("base_query_overlap_checked") or doc.get("normalization") != NORMALIZATION:
        raise ValueError("conversion missing required normalization/overlap checks")
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    for stage in ("probe", "download", "lock", "convert"):
        x = sub.add_parser(stage)
        x.add_argument("--profile", choices=PROFILES, required=True)
        x.add_argument("--out", required=True)
        if stage in ("probe", "lock"):
            x.add_argument("--raw-dir", required=stage == "lock")
        if stage == "convert":
            x.add_argument("--source-lock", required=True)
            x.add_argument("--normalize", choices=[NORMALIZATION], default=NORMALIZATION)
            x.add_argument("--batch-size", type=int, default=4096)
            x.add_argument("--holdout-count", type=int)
            x.add_argument("--holdout-seed", type=int, default=20261010)
            x.add_argument("--allow-synthetic", action="store_true")
    args = p.parse_args()
    {"probe": probe, "download": download, "lock": lock_local, "convert": convert}[args.stage](args)


if __name__ == "__main__":
    cli(main)
