#!/usr/bin/env bash
#SBATCH --job-name=hnsw_index
#SBATCH --account=db4ai
#SBATCH --partition=db4ai
#SBATCH --qos=db4ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
: "${BUILD_OFF:?}" "${DATASET_CONFIG:?}" "${INDEX_PATH:?}" "${RUN_DIR:?}" "${REPO_ROOT:?}"
mkdir -p "$(dirname "$INDEX_PATH")" "$RUN_DIR/index"
if [[ -f "$INDEX_PATH" && -f "$INDEX_PATH.manifest.json" ]]; then
  python3 - "$INDEX_PATH" <<'PY'
import hashlib, json, pathlib, sys
index = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(str(index) + ".manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = manifest.get("index_sha256")
if expected:
    digest = hashlib.sha256()
    with index.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit("Existing index checksum does not match its manifest")
print("existing_index_integrity_ok")
PY
  echo "Reusing validated existing index: $INDEX_PATH"
  exit 0
fi
if [[ -e "$INDEX_PATH" || -e "$INDEX_PATH.manifest.json" ]]; then
  echo "Index and manifest are not both present; refusing ambiguous reuse" >&2
  exit 1
fi
"$BUILD_OFF/real_data_trace_runner" \
  --dataset-config "$DATASET_CONFIG" --mode build-index \
  --index-path "$INDEX_PATH" --output-dir "$RUN_DIR/index" --run-id "$RUN_ID" \
  --M "$M" --ef-construction "$EF_CONSTRUCTION" --seed "$SEED" \
  --threads "$SLURM_CPUS_PER_TASK"
python3 - "$INDEX_PATH" "$REPO_ROOT" <<'PY'
import hashlib, json, pathlib, subprocess, sys
index = pathlib.Path(sys.argv[1])
repo = pathlib.Path(sys.argv[2])
manifest_path = pathlib.Path(str(index) + ".manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
digest = hashlib.sha256()
with index.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
manifest.update({
    "index_bytes": index.stat().st_size,
    "index_sha256": digest.hexdigest(),
    "git_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True).strip()),
    "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
    "build_type": "Release",
})
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("index_manifest_enriched")
PY
echo "build_index_job_ok"
