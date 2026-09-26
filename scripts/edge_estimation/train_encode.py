from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.edge_estimation.contracts import atomic_output_dir, file_entry, load_strict_json, sha256_file
    from scripts.edge_estimation.capabilities import require_training_backend
else:
    from .contracts import atomic_output_dir, file_entry, load_strict_json, sha256_file
    from .capabilities import require_training_backend


def read_fvecs(path: Path, dimension: int) -> np.ndarray:
    raw = np.memmap(path, dtype="<i4", mode="r")
    stride = dimension + 1
    if raw.size % stride:
        raise ValueError("fvecs size is not a whole number of rows")
    rows = raw.reshape(-1, stride)
    if not np.all(rows[:, 0] == dimension):
        raise ValueError("fvecs dimension mismatch")
    return rows[:, 1:].view("<f4")


def read_catalog(directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = load_strict_json(directory / "manifest.json")
    offsets = np.fromfile(directory / "source_offsets.u64le", dtype="<u8")
    targets = np.fromfile(directory / "targets.u32le", dtype="<u4")
    if offsets.size != int(manifest["node_count"]) + 1 or offsets[0] != 0 or offsets[-1] != targets.size:
        raise ValueError("catalog offsets mismatch")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("catalog offsets are non-monotonic")
    return offsets, targets, manifest


def directions(base: np.ndarray, offsets: np.ndarray, targets: np.ndarray,
               edge_ids: np.ndarray) -> np.ndarray:
    sources = np.searchsorted(offsets[1:], edge_ids, side="right")
    delta = base[targets[edge_ids].astype(np.int64)].astype(np.float64) - base[sources].astype(np.float64)
    lengths = np.linalg.norm(delta, axis=1)
    result = np.zeros_like(delta, dtype=np.float32)
    valid = lengths > 0
    result[valid] = (delta[valid] / lengths[valid, None]).astype(np.float32)
    return result


def kmeans(values: np.ndarray, k: int, iterations: int, seed: int) -> np.ndarray:
    if values.shape[0] < k:
        raise ValueError(f"training sample {values.shape[0]} is smaller than centroid count {k}")
    rng = np.random.default_rng(seed)
    centers = values[rng.choice(values.shape[0], k, replace=False)].copy()
    for _ in range(iterations):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignment = distances.argmin(axis=1)
        for index in range(k):
            members = values[assignment == index]
            if members.size:
                centers[index] = members.mean(axis=0)
    return centers.astype("<f4")


def pack_codes(codes: np.ndarray, nbits: int) -> bytes:
    output = bytearray((codes.size * nbits + 7) // 8)
    for index, value in enumerate(codes.tolist()):
        bit = index * nbits
        output[bit // 8] |= (value << (bit % 8)) & 0xFF
        if bit % 8 + nbits > 8:
            output[bit // 8 + 1] |= value >> (8 - bit % 8)
    return bytes(output)


def train_encode(config_path: Path, assets_path: Path, catalog_dir: Path, output: Path) -> None:
    config = load_strict_json(config_path)
    assets = load_strict_json(assets_path)
    codec = config["codec"]
    require_training_backend(str(codec.get("kind")))
    if codec.get("kind") != "pq_packed":
        raise RuntimeError(f"backend unavailable for train-encode: {codec.get('kind')}")
    dimension = int(config.get("capture", {}).get("dimension", 0))
    m, nbits = int(codec["m"]), int(codec["nbits"])
    if dimension <= 0 or m <= 0 or dimension % m or not 1 <= nbits <= 8:
        raise ValueError("packed PQ requires dimension divisible by m and nbits 1..8")
    dsub, ksub = dimension // m, 1 << nbits
    offsets, targets, catalog = read_catalog(catalog_dir)
    base = read_fvecs(Path(assets["base"]), dimension)
    if base.shape[0] != offsets.size - 1:
        raise ValueError("base vector count does not match catalog")
    edge_count = targets.size
    if edge_count == 0:
        raise ValueError("cannot train packed PQ on an empty edge catalog")
    trainer = config.get("trainer", {})
    sample_cap = min(edge_count, int(trainer.get("sample_cap", min(edge_count, 20000))))
    seed, iterations = int(trainer.get("seed", 20260924)), int(trainer.get("iterations", 12))
    rng = np.random.default_rng(seed)
    sample_ids = np.sort(rng.choice(edge_count, sample_cap, replace=False))
    sample = directions(base, offsets, targets, sample_ids)
    codebooks = np.empty((m, ksub, dsub), dtype="<f4")
    for sub in range(m):
        codebooks[sub] = kmeans(sample[:, sub * dsub:(sub + 1) * dsub], ksub, iterations, seed + sub)
    code_bytes = (m * nbits + 7) // 8
    record_size = 16 + code_bytes
    with atomic_output_dir(output) as partial:
        model_dir, records_dir = partial / "model", partial / "records"
        model_dir.mkdir(); records_dir.mkdir()
        codebook_path, records_path = model_dir / "codebook.f32le", records_dir / "edges.bin"
        codebooks.tofile(codebook_path)
        with records_path.open("wb") as stream:
            for source in range(base.shape[0]):
                first, last = int(offsets[source]), int(offsets[source + 1])
                for edge_id in range(first, last):
                    delta = base[int(targets[edge_id])].astype(np.float64) - base[source].astype(np.float64)
                    length = float(np.linalg.norm(delta))
                    unit = np.zeros(dimension, dtype=np.float32) if length == 0 else (delta / length).astype(np.float32)
                    codes = np.empty(m, dtype=np.uint8)
                    reconstructed = np.empty(dimension, dtype=np.float32)
                    for sub in range(m):
                        part = unit[sub * dsub:(sub + 1) * dsub]
                        code = int(((codebooks[sub] - part) ** 2).sum(axis=1).argmin())
                        codes[sub] = code
                        reconstructed[sub * dsub:(sub + 1) * dsub] = codebooks[sub, code]
                    anchor = float(np.dot(base[source].astype(np.float64), reconstructed.astype(np.float64)))
                    stream.write(struct.pack("<dd", length, anchor))
                    stream.write(pack_codes(codes, nbits))
        native = partial / "native.cfg"
        codebook_sha256 = sha256_file(codebook_path)
        records_sha256 = sha256_file(records_path)
        native.write_text("\n".join([
            "format=uq-pq-packed/1", f"dimension={dimension}", f"m={m}", f"nbits={nbits}",
            f"dsub={dsub}", f"edge_count={edge_count}", f"record_size={record_size}",
            f"catalog_identity={catalog['identity_sha256']}", "coverage=full_graph",
            "codebook=model/codebook.f32le", f"codebook_sha256={codebook_sha256}",
            "records=records/edges.bin", f"records_sha256={records_sha256}", ""
        ]), encoding="ascii")
        training = partial / "training.json"
        training.write_text(json.dumps({"seed": seed, "iterations": iterations,
            "sample_cap": sample_cap,
            "sample_ids_sha256": hashlib.sha256(sample_ids.astype('<u8').tobytes()).hexdigest(),
            "libraries": {"python": sys.version.split()[0], "numpy": np.__version__}},
            indent=2) + "\n")
        manifest = partial / "manifest.json"
        def artifact_entry(path: Path, relative: str) -> dict:
            return {**file_entry(path), "path": relative}
        output_files = {
            "codebook": artifact_entry(codebook_path, "model/codebook.f32le"),
            "records": artifact_entry(records_path, "records/edges.bin"),
            "native_config": artifact_entry(native, "native.cfg"),
            "training": artifact_entry(training, "training.json"),
        }
        inputs = {
            "config": file_entry(config_path),
            "assets_contract": file_entry(assets_path),
            "base": file_entry(Path(assets["base"])),
            "catalog_manifest": file_entry(catalog_dir / "manifest.json"),
        }
        manifest.write_text(json.dumps({"schema_version": 1, "format_family": "uq-pq-packed",
            "format_version": 1, "representation": "direct_unit_edge", "codec": codec,
            "correction": config["correction"], "numeric_profile": "float_lut_v1",
            "dimension": dimension, "edge_catalog_identity": catalog["identity_sha256"],
            "coverage": "full_graph", "edge_count": int(edge_count),
            "resolved_config": config, "inputs": inputs, "files": output_files},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "complete.json").write_text(json.dumps({"schema_version": 1,
            "stage": "train-encode", "inputs": inputs,
            "outputs": {**output_files, "manifest": file_entry(manifest)}},
            indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--assets", required=True)
    parser.add_argument("--catalog", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    train_encode(Path(args.config), Path(args.assets), Path(args.catalog), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
