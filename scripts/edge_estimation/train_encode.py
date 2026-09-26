from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.edge_estimation.contracts import (
        atomic_output_dir, file_entry, load_strict_json, sha256_file,
        validate_config_v1,
    )
    from scripts.edge_estimation.trainers import train_product_model
else:
    from .contracts import (atomic_output_dir, file_entry, load_strict_json,
                            sha256_file, validate_config_v1)
    from .trainers import train_product_model


def read_fvecs(path: Path, dimension: int) -> np.ndarray:
    raw = np.memmap(path, dtype="<i4", mode="r")
    stride = dimension + 1
    if raw.size % stride:
        raise ValueError("fvecs size is not a whole number of rows")
    rows = raw.reshape(-1, stride)
    if not np.all(rows[:, 0] == dimension):
        raise ValueError("fvecs dimension mismatch")
    values = rows[:, 1:].view("<f4")
    if not np.isfinite(values).all():
        raise ValueError("base vectors contain non-finite values")
    return values


def read_catalog(directory: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = load_strict_json(directory / "manifest.json")
    offsets = np.fromfile(directory / "source_offsets.u64le", dtype="<u8")
    targets = np.fromfile(directory / "targets.u32le", dtype="<u4")
    if (offsets.size != int(manifest["node_count"]) + 1 or offsets[0] != 0 or
            offsets[-1] != targets.size):
        raise ValueError("catalog offsets mismatch")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("catalog offsets are non-monotonic")
    if targets.size and int(targets.max()) >= int(manifest["node_count"]):
        raise ValueError("catalog target is outside the internal-ID range")
    return offsets, targets, manifest


def load_internal_to_base(assets: dict[str, Any], node_count: int,
                          base_count: int) -> tuple[np.ndarray, str, dict[str, Any]]:
    raw_path = assets.get("internal_to_label")
    if raw_path is None:
        if node_count != base_count:
            raise ValueError("internal_to_label is required when catalog and base counts differ")
        mapping = np.arange(node_count, dtype=np.uint64)
        identity = hashlib.sha256(mapping.astype("<u8").tobytes()).hexdigest()
        return mapping, identity, {"kind": "verified_identity", "count": node_count}
    path = Path(str(raw_path))
    mapping = np.load(path, mmap_mode="r", allow_pickle=False)
    if mapping.ndim != 1 or mapping.size != node_count or mapping.dtype.kind not in "iu":
        raise ValueError("internal_to_label must be a one-dimensional integer .npy array")
    if mapping.size and (int(mapping.min()) < 0 or int(mapping.max()) >= base_count):
        raise ValueError("internal_to_label contains a base row outside the available range")
    normalized = np.array(mapping, dtype=np.uint64, copy=True)
    del mapping
    if np.unique(normalized).size != node_count:
        raise ValueError("internal_to_label must be one-to-one")
    return normalized, sha256_file(path), {
        "kind": "npy_internal_to_external_label", "path": str(path.resolve()),
        "file": file_entry(path), "count": node_count,
    }


def edge_geometry(base: np.ndarray, mapping: np.ndarray, offsets: np.ndarray,
                  targets: np.ndarray, edge_ids: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources = np.searchsorted(offsets[1:], edge_ids, side="right")
    source_rows = mapping[sources].astype(np.int64, copy=False)
    target_rows = mapping[targets[edge_ids].astype(np.int64)].astype(np.int64, copy=False)
    delta = (base[target_rows].astype(np.float64) -
             base[source_rows].astype(np.float64))
    lengths = np.linalg.norm(delta, axis=1)
    unit = np.zeros_like(delta, dtype=np.float32)
    valid = np.isfinite(lengths) & (lengths > 0.0)
    unit[valid] = (delta[valid] / lengths[valid, None]).astype(np.float32)
    return sources, source_rows, lengths, unit


def pack_code_matrix(codes: np.ndarray, nbits: int) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint16)
    if values.ndim != 2 or not 1 <= nbits <= 8:
        raise ValueError("invalid canonical product codes")
    if np.any(values >= (1 << nbits)):
        raise ValueError("product code does not fit configured nbits")
    code_bytes = (values.shape[1] * nbits + 7) // 8
    packed = np.zeros((values.shape[0], code_bytes), dtype=np.uint8)
    for sub in range(values.shape[1]):
        bit = sub * nbits
        byte, shift = divmod(bit, 8)
        packed[:, byte] |= ((values[:, sub] << shift) & 0xFF).astype(np.uint8)
        if shift + nbits > 8:
            packed[:, byte + 1] |= (values[:, sub] >> (8 - shift)).astype(np.uint8)
    return packed


def _artifact_entry(path: Path, relative: str) -> dict[str, Any]:
    return {**file_entry(path), "path": relative}


def _write_records(path: Path, model: Any, base: np.ndarray,
                   mapping: np.ndarray, offsets: np.ndarray, targets: np.ndarray,
                   batch_size: int) -> tuple[int, int]:
    edge_count = int(targets.size)
    code_bytes = (int(model.code_size) if getattr(model, "native_code_bytes", False)
                  else (model.m * model.nbits + 7) // 8)
    record_size = 16 + code_bytes
    with path.open("wb") as stream:
        for first in range(0, edge_count, batch_size):
            last = min(edge_count, first + batch_size)
            edge_ids = np.arange(first, last, dtype=np.int64)
            _, source_rows, lengths, unit = edge_geometry(
                base, mapping, offsets, targets, edge_ids)
            codes, _, reconstructed = model.encode_batch(unit)
            source_vectors = np.asarray(base[source_rows], dtype=np.float32)
            source_model_space = model.transform(source_vectors)
            anchors = np.einsum(
                "ij,ij->i", source_model_space.astype(np.float64),
                reconstructed.astype(np.float64))
            packed = (np.asarray(codes, dtype=np.uint8) if
                      getattr(model, "native_code_bytes", False) else
                      pack_code_matrix(codes, model.nbits))
            payload = bytearray((last - first) * record_size)
            np.ndarray((last - first,), dtype="<f8", buffer=payload,
                       offset=0, strides=(record_size,))[:] = lengths
            np.ndarray((last - first,), dtype="<f8", buffer=payload,
                       offset=8, strides=(record_size,))[:] = anchors
            np.ndarray((last - first, code_bytes), dtype="u1", buffer=payload,
                       offset=16, strides=(record_size, 1))[:] = packed
            stream.write(payload)
    return record_size, code_bytes


def train_encode(config_path: Path, assets_path: Path, catalog_dir: Path,
                 output: Path) -> None:
    config = load_strict_json(config_path)
    validate_config_v1(config)
    assets = load_strict_json(assets_path)
    codec = dict(config["codec"])
    kind = str(codec.get("kind"))
    if kind not in {"pq_packed", "opq", "prq", "jq", "rabitq"}:
        raise RuntimeError(f"backend unavailable for train-encode: {kind}")
    dimension = int(config.get("capture", {}).get("dimension", 0))
    nbits = int(codec.get("nbits", 8))
    if kind == "prq":
        nsplits = int(codec.get("nsplits", 16))
        stages = int(codec.get("stages_per_split", 2))
        m = nsplits * stages
        divisor = nsplits
        if stages <= 0:
            raise ValueError("PRQ stages_per_split must be positive")
        codec["nsplits"], codec["stages_per_split"] = nsplits, stages
    elif kind == "rabitq":
        m = dimension
        divisor = dimension
        nbits = 1
        codec["nbits"] = 1
    else:
        m = int(codec.get("m", 32 if kind in {"opq", "jq"} else 0))
        divisor = m
        codec["m"] = m
    if dimension <= 0 or divisor <= 0 or dimension % divisor or not 1 <= nbits <= 8:
        raise ValueError("quantizer shape is incompatible with dimension or nbits")
    codec["nbits"] = nbits
    offsets, targets, catalog = read_catalog(catalog_dir)
    base = read_fvecs(Path(assets["base"]), dimension)
    node_count = offsets.size - 1
    mapping, mapping_identity, mapping_metadata = load_internal_to_base(
        assets, node_count, base.shape[0])
    edge_count = int(targets.size)
    if edge_count == 0:
        raise ValueError("cannot train on an empty edge catalog")
    trainer = dict(config.get("trainer", {}))
    trainer.setdefault("provider", "behavioral_port" if kind == "jq" else
                       "faiss" if kind in {"opq", "rabitq"} else "numpy_reference")
    sample_cap = min(edge_count, int(trainer.get("sample_cap", min(edge_count, 20000))))
    seed = int(trainer.get("seed", 20260924))
    rng = np.random.default_rng(seed)
    candidate_ids = np.sort(rng.choice(edge_count, sample_cap, replace=False))
    _, _, sample_lengths, sample = edge_geometry(
        base, mapping, offsets, targets, candidate_ids)
    valid_sample = np.isfinite(sample_lengths) & (sample_lengths > 0.0)
    sample_ids = candidate_ids[valid_sample]
    sample = sample[valid_sample]
    if sample.shape[0] < (1 << nbits):
        raise ValueError("valid non-zero training sample is smaller than centroid count")

    training_started = time.perf_counter()
    model = train_product_model(codec, trainer, sample)
    training_seconds = time.perf_counter() - training_started
    if model.dimension != dimension or model.m != m or model.nbits != nbits:
        raise RuntimeError("trainer returned a model with the wrong shape")

    batch_size = int(trainer.get("encode_batch_size", 4096))
    if batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")
    with atomic_output_dir(output) as partial:
        model_dir, records_dir = partial / "model", partial / "records"
        model_dir.mkdir(); records_dir.mkdir()
        codebook_path = model_dir / "codebook.f32le"
        records_path = records_dir / "edges.bin"
        sample_ids_path = partial / "training_sample_ids.u64le"
        np.ascontiguousarray(model.codebooks, dtype="<f4").tofile(codebook_path)
        sample_ids.astype("<u8").tofile(sample_ids_path)
        rotation_path: Path | None = None
        if model.rotation is not None:
            rotation_path = model_dir / "rotation.f32le"
            np.ascontiguousarray(model.rotation, dtype="<f4").tofile(rotation_path)
        encoding_started = time.perf_counter()
        record_size, code_bytes = _write_records(
            records_path, model, base, mapping, offsets, targets, batch_size)
        encoding_seconds = time.perf_counter() - encoding_started

        codebook_sha256 = sha256_file(codebook_path)
        records_sha256 = sha256_file(records_path)
        native_lines = [
            f"format={('uq-rotated-pq/1' if kind in {'opq', 'jq'} else 'uq-prq/1' if kind == 'prq' else 'uq-rabitq/1' if kind == 'rabitq' else 'uq-pq-packed/1')}",
            f"backend={kind}", f"implementation={model.implementation}",
            f"provider={model.provider}", f"dimension={dimension}", f"m={m}",
            f"nbits={nbits}", f"dsub={dimension // divisor}", f"edge_count={edge_count}",
            f"record_size={record_size}", f"catalog_identity={catalog['identity_sha256']}",
            f"mapping_identity={mapping_identity}", "coverage=full_graph",
            "codebook=model/codebook.f32le", f"codebook_sha256={codebook_sha256}",
            "records=records/edges.bin", f"records_sha256={records_sha256}",
        ]
        if kind == "prq":
            native_lines.extend([
                f"nsplits={nsplits}", f"stages_per_split={stages}", "norm_bits=0",
                "codebook_layout=split_stage_centroid_dimension",
            ])
        if kind == "rabitq":
            native_lines.extend([
                f"code_size={model.code_size}", f"sign_bytes={(dimension + 7) // 8}",
                "metric=inner_product", "query_bits=0", "centroid=zero",
            ])
        if rotation_path is not None:
            native_lines.extend([
                "rotation=model/rotation.f32le",
                f"rotation_sha256={sha256_file(rotation_path)}",
                "rotation_layout=row_major_r_times_column", "rotation_bias=none",
            ])
        native = partial / "native.cfg"
        native.write_text("\n".join(native_lines) + "\n", encoding="ascii")

        training = partial / "training.json"
        training_payload = {
            "schema_version": 1, "backend": kind,
            "implementation": model.implementation, "provider": model.provider,
            "seed": seed, "requested_sample_cap": sample_cap,
            "valid_sample_count": int(sample_ids.size),
            "sample_ids_sha256": sha256_file(sample_ids_path),
            "parameters": trainer, "training_seconds": training_seconds,
            "encoding_seconds": encoding_seconds,
            "dependency_versions": model.dependency_versions or {},
            "python": sys.version.split()[0], "mapping": mapping_metadata,
        }
        training.write_text(json.dumps(training_payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        output_files: dict[str, Any] = {
            "codebook": _artifact_entry(codebook_path, "model/codebook.f32le"),
            "records": _artifact_entry(records_path, "records/edges.bin"),
            "native_config": _artifact_entry(native, "native.cfg"),
            "training": _artifact_entry(training, "training.json"),
            "training_sample_ids": _artifact_entry(
                sample_ids_path, "training_sample_ids.u64le"),
        }
        if rotation_path is not None:
            output_files["rotation"] = _artifact_entry(rotation_path, "model/rotation.f32le")
        inputs = {
            "config": file_entry(config_path), "assets_contract": file_entry(assets_path),
            "base": file_entry(Path(assets["base"])),
            "catalog_manifest": file_entry(catalog_dir / "manifest.json"),
        }
        if "internal_to_label" in assets:
            inputs["internal_to_label"] = file_entry(Path(assets["internal_to_label"]))
        manifest = partial / "manifest.json"
        manifest_payload = {
            "schema_version": 1,
            "format_family": ("uq-rotated-pq" if kind in {"opq", "jq"} else
                              "uq-prq" if kind == "prq" else
                              "uq-rabitq" if kind == "rabitq" else "uq-pq-packed"),
            "format_version": 1,
            "algorithm": kind, "implementation": model.implementation,
            "provider": model.provider, "representation": config["representation"],
            "codec": codec, "correction": config["correction"],
            "numeric_profile": "float_lut_v1", "dimension": dimension,
            "edge_catalog_identity": catalog["identity_sha256"],
            "internal_id_mapping_identity": mapping_identity,
            "coverage": "full_graph", "edge_count": edge_count,
            "record_stride": record_size, "payload_bytes": code_bytes,
            "endian": "little", "resolved_config": config,
            "dependency_versions": model.dependency_versions or {},
            "inputs": inputs, "files": output_files,
        }
        manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        complete_outputs = {**output_files, "manifest": file_entry(manifest)}
        (partial / "complete.json").write_text(json.dumps({
            "schema_version": 1, "stage": "train-encode", "inputs": inputs,
            "outputs": complete_outputs,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    train_encode(Path(args.config), Path(args.assets), Path(args.catalog), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
