#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${1:-$HOME/IndividualProject/datasets/sift1m}"
URL="${SIFT1M_URL:-ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz}"
ARCHIVE="$DATASET_DIR/sift.tar.gz"

mkdir -p "$DATASET_DIR"

if [[ ! -f "$ARCHIVE" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$ARCHIVE.part" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="$ARCHIVE.part" "$URL"
  else
    echo "Neither curl nor wget is available" >&2
    exit 1
  fi
  mv "$ARCHIVE.part" "$ARCHIVE"
fi

if [[ ! -f "$DATASET_DIR/sift/sift_base.fvecs" ]]; then
  tar -xzf "$ARCHIVE" -C "$DATASET_DIR"
fi

for required in sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs; do
  if [[ ! -s "$DATASET_DIR/sift/$required" ]]; then
    echo "Missing extracted SIFT1M file: $DATASET_DIR/sift/$required" >&2
    exit 1
  fi
done

python3 - "$DATASET_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import struct
import sys

root = Path(sys.argv[1]).resolve()
source = root / "sift"

def inspect(path: Path, component_bytes: int) -> tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(4)
    if len(raw) != 4:
        raise RuntimeError(f"empty vector file: {path}")
    dimension = struct.unpack("<i", raw)[0]
    if dimension <= 0:
        raise RuntimeError(f"invalid dimension in {path}")
    record_bytes = 4 + dimension * component_bytes
    if size % record_bytes:
        raise RuntimeError(f"truncated or malformed vector file: {path}")
    return dimension, size // record_bytes

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

base = source / "sift_base.fvecs"
query = source / "sift_query.fvecs"
truth = source / "sift_groundtruth.ivecs"
dimension, n_base = inspect(base, 4)
query_dimension, n_query = inspect(query, 4)
ground_truth_k, ground_truth_queries = inspect(truth, 4)
if dimension != query_dimension or n_query != ground_truth_queries:
    raise RuntimeError("SIFT1M file dimensions or query counts disagree")

manifest = {
    "dataset": "sift1m",
    "distance_kind": "squared_l2_float32",
    "base_path": str(base),
    "query_path": str(query),
    "ground_truth_path": str(truth),
    "base_format": "fvecs",
    "query_format": "fvecs",
    "ground_truth_format": "ivecs",
    "dimension": dimension,
    "n_base": n_base,
    "n_query": n_query,
    "ground_truth_k": ground_truth_k,
    "checksums": {
        "base_sha256": sha256(base),
        "query_sha256": sha256(query),
        "ground_truth_sha256": sha256(truth),
    },
}
(root / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
print("prepare_sift1m_ok")
PY
