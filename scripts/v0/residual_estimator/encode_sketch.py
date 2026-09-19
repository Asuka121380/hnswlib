"""Encode one shared Gaussian matrix per seed and unique frozen edge."""

import argparse
import struct
from pathlib import Path

import numpy as np

from common import mark_complete, read_json, sha256, write_json


def edge_dtype(dimension: int, pq_m: int) -> np.dtype:
    return np.dtype([
        ("ordinal", "<u8"), ("source", "<u4"), ("target", "<u4"),
        ("slot", "<u4"), ("flags", "<u4"), ("length", "<f8"),
        ("anchor", "<f8"), ("bias", "<f8"),
        ("centroid", "<f4", (dimension,)),
        ("residual", "<f8", (dimension,)),
        ("code", "u1", (pq_m,)),
    ], align=False)


def open_edges(path: str | Path, dimension: int) -> np.memmap:
    with Path(path).open("rb") as source:
        magic = source.read(8)
        stored_dimension, count, pq_m = struct.unpack("<IQI", source.read(16))
    if magic != b"V0RGE002" or stored_dimension != dimension or not pq_m:
        raise ValueError("edge geometry header mismatch")
    dtype = edge_dtype(dimension, pq_m)
    if Path(path).stat().st_size != 24 + count * dtype.itemsize:
        raise ValueError("edge geometry byte count mismatch")
    return np.memmap(path, mode="r", offset=24, dtype=dtype, shape=(count,))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    dataset, contract = read_json(args.dataset), read_json(args.contract)
    config = contract["config"]
    dimension = int(config["dimension"])
    edges = open_edges(dataset["edges_bin"], dimension)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "dataset": str(Path(args.dataset).resolve()),
              "contract": str(Path(args.contract).resolve()),
              "edge_count": len(edges), "sketches": []}
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    for seed in config["projection_seeds"]:
        rng = np.random.default_rng(seed)
        maximum = max(config["residual_bits"])
        reference = rng.standard_normal((maximum, dimension), dtype=np.float64)
        fast = reference.astype("<f4")
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir(exist_ok=True)
        np.save(seed_dir / "G_ref.f64.npy", reference)
        np.save(seed_dir / "G_fast.f32.npy", fast)
        for bits in config["residual_bits"]:
            if bits % 8 or bits > maximum:
                raise ValueError("invalid residual code length")
            target = seed_dir / f"k-{bits}"
            target.mkdir(exist_ok=True)
            matrix = fast[:bits].copy()
            matrix.tofile(target / "matrix.f32")
            packed = np.lib.format.open_memmap(
                target / "signs.npy", mode="w+", dtype="u1",
                shape=(len(edges), bits // 8))
            scales = np.lib.format.open_memmap(
                target / "scales.npy", mode="w+", dtype="<f4", shape=(len(edges),))
            offsets = np.lib.format.open_memmap(
                target / "offsets.npy", mode="w+", dtype="<f4", shape=(len(edges),))
            valid = np.lib.format.open_memmap(
                target / "valid.npy", mode="w+", dtype="u1", shape=(len(edges),))
            gaussian = matrix.astype(np.float64)
            for start in range(0, len(edges), args.batch_size):
                stop = min(start + args.batch_size, len(edges))
                batch = edges[start:stop]
                z = np.asarray(batch["residual"], dtype=np.float64)
                c = np.asarray(batch["centroid"], dtype=np.float64)
                projected = z @ gaussian.T
                signs = projected >= 0.0
                packed[start:stop] = np.packbits(signs, axis=1, bitorder="little")
                scale = 2.0 * np.linalg.norm(z, axis=1) * np.sqrt(np.pi / 2.0) / bits
                centroid_projection = c @ gaussian.T
                signed_centroid = np.sum(
                    np.where(signs, centroid_projection, -centroid_projection), axis=1)
                offset = scale * signed_centroid - batch["bias"]
                good = ((batch["flags"] & 7) == 0) & np.isfinite(scale) & np.isfinite(offset)
                good &= (np.abs(scale) <= np.finfo("f4").max)
                good &= (np.abs(offset) <= np.finfo("f4").max)
                valid[start:stop] = good.astype("u1")
                scales[start:stop] = np.where(good, scale, 0).astype("<f4")
                offsets[start:stop] = np.where(good, offset, 0).astype("<f4")
            for array in (packed, scales, offsets, valid):
                array.flush()
            item = {"seed": seed, "bits": bits, "directory": str(target.resolve()),
                    "matrix_sha256": sha256(target / "matrix.f32"),
                    "valid_count": int(np.count_nonzero(valid))}
            result["sketches"].append(item)
    write_json(output / "manifest.json", result)
    mark_complete(output, {"dataset": sha256(args.dataset),
                           "contract": sha256(args.contract)},
                  {"manifest": sha256(output / "manifest.json")})


if __name__ == "__main__":
    main()
