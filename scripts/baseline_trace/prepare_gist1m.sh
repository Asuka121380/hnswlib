#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASET_DIR="${1:-$HOME/IndividualProject/datasets/gist1m}"
URL="${GIST1M_URL:-ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz}"
ARCHIVE="$DATASET_DIR/gist.tar.gz"
MANIFEST="$DATASET_DIR/dataset.json"
COMPLETE="$DATASET_DIR/complete.json"
KEEP_ARCHIVE="${GIST1M_KEEP_ARCHIVE:-0}"

mkdir -p "$DATASET_DIR"

if [[ -f "$COMPLETE" && -f "$MANIFEST" ]]; then
  python3 "$SCRIPT_DIR/validate_dataset.py" --dataset-config "$MANIFEST"
  echo "prepare_gist1m_already_complete dataset_config=$MANIFEST"
  exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
  rm -f "$ARCHIVE.part"
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

if [[ ! -f "$DATASET_DIR/gist/gist_base.fvecs" ]]; then
  tar -xzf "$ARCHIVE" -C "$DATASET_DIR"
fi

for required in gist_base.fvecs gist_query.fvecs gist_groundtruth.ivecs; do
  if [[ ! -s "$DATASET_DIR/gist/$required" ]]; then
    echo "Missing extracted GIST1M file: $DATASET_DIR/gist/$required" >&2
    exit 1
  fi
done

TEMP_MANIFEST="$DATASET_DIR/.dataset.json.tmp-$$"
python3 - "$DATASET_DIR" "$URL" "$TEMP_MANIFEST" <<'PY'
import hashlib
import json
from pathlib import Path
import struct
import sys

root = Path(sys.argv[1]).resolve()
source_url = sys.argv[2]
manifest_path = Path(sys.argv[3])
source = root / "gist"


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


base = source / "gist_base.fvecs"
query = source / "gist_query.fvecs"
truth = source / "gist_groundtruth.ivecs"
dimension, n_base = inspect(base, 4)
query_dimension, n_query = inspect(query, 4)
ground_truth_k, ground_truth_queries = inspect(truth, 4)
expected = (dimension, n_base, n_query, ground_truth_k)
if expected != (960, 1_000_000, 1_000, 100):
    raise RuntimeError(f"unexpected GIST1M shape: {expected}")
if query_dimension != dimension or ground_truth_queries != n_query:
    raise RuntimeError("GIST1M file dimensions or query counts disagree")

manifest = {
    "dataset": "gist1m",
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
    "source": {"url": source_url},
    "file_sizes": {
        "base_bytes": base.stat().st_size,
        "query_bytes": query.stat().st_size,
        "ground_truth_bytes": truth.stat().st_size,
    },
    "checksums": {
        "base_sha256": sha256(base),
        "query_sha256": sha256(query),
        "ground_truth_sha256": sha256(truth),
    },
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

python3 "$SCRIPT_DIR/validate_dataset.py" --dataset-config "$TEMP_MANIFEST"
mv "$TEMP_MANIFEST" "$MANIFEST"

python3 - "$COMPLETE" "$MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    json.dumps({"status": "complete", "dataset": "gist1m", "dataset_config": sys.argv[2]}, indent=2) + "\n",
    encoding="utf-8",
)
PY

if [[ "$KEEP_ARCHIVE" != "1" ]]; then
  rm -f "$ARCHIVE"
fi

echo "prepare_gist1m_ok dataset_config=$MANIFEST"
