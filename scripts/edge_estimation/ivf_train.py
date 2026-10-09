"""Coarse residual *edge* quantization; intentionally no IVF list routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.edge_estimation.contracts import atomic_output_dir, sha256_file
from scripts.edge_estimation.train_encode import (
    read_fvecs, read_catalog, load_internal_to_base, edge_geometry, pack_code_matrix)
from scripts.edge_estimation.trainers import train_product_model


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inventory(root):
    return {str(p.relative_to(root)).replace("\\", "/"): {
        "sha256": sha256_file(p), "size": p.stat().st_size}
        for p in sorted(root.rglob("*")) if p.is_file() and
        p.name not in {"manifest.json", "complete.json", "native.cfg"}}


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    if complete["manifest_sha256"] != sha256_file(root / "manifest.json"):
        raise ValueError("manifest identity mismatch")
    for name, entry in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("artifact path escape")
        if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"artifact hash/size mismatch: {name}")
    if "native_sha256" in complete and sha256_file(root / "native.cfg") != complete["native_sha256"]:
        raise ValueError("native config identity mismatch")
    return manifest


def context(assets_path, catalog_dir, dimension):
    assets = json.loads(Path(assets_path).read_text())
    if not assets.get("internal_to_label"):
        raise ValueError("explicit internal_to_label .npy required, including identity mappings")
    offsets, targets, cat = read_catalog(Path(catalog_dir))
    cat["catalog_identity"] = cat["identity_sha256"]
    # Same canonical digest as EdgeCatalog; verify content rather than trusting a label.
    digest = hashlib.sha256(b"UQCAT001" + struct.pack("<QQ", len(offsets)-1, len(targets)) +
                            bytes.fromhex(cat["adjacency_sha256"]) +
                            offsets.astype("<u8").tobytes() + targets.astype("<u4").tobytes()).hexdigest()
    if digest != cat["catalog_identity"]:
        raise ValueError("catalog content identity mismatch")
    base = read_fvecs(Path(assets["base"]), dimension)
    mapping, _, _ = load_internal_to_base(assets, len(offsets)-1, len(base))
    hashes = {key: sha256_file(Path(assets[key])) for key in ("base", "index", "internal_to_label")}
    complete=json.loads((Path(catalog_dir)/"complete.json").read_text())
    if complete.get("index_sha256") != hashes["index"]:
        raise ValueError("catalog was not exported from the supplied index")
    # hnswlib's persisted 64-bit little-endian layout (saveIndex in hnswalg.h).
    # Prove vector correspondence; hashes of unrelated files alone are insufficient.
    header = struct.Struct("<6QiI3QdQ")
    with Path(assets["index"]).open("rb") as stream:
        fields=header.unpack(stream.read(header.size))
    _, capacity, count, stride, label_offset, data_offset, *_ = fields
    if (count != len(mapping) or capacity<count or data_offset+4*dimension != label_offset or
            label_offset+8 != stride or header.size+count*stride > Path(assets["index"]).stat().st_size):
        raise ValueError("unsupported HNSW serialization or dimension/count mismatch (requires 64-bit LE)")
    raw=np.memmap(assets["index"],dtype="u1",mode="r")
    stored=np.ndarray((count,dimension),dtype="<f4",buffer=raw,offset=header.size+data_offset,strides=(stride,4))
    for first in range(0,count,4096):
        last=min(count,first+4096)
        if not np.array_equal(stored[first:last],base[mapping[first:last].astype(np.int64)]):
            raise ValueError(f"base/mapping vectors differ from index at internal-ID block {first}:{last}")
    del stored,raw
    return base, mapping, offsets, targets, cat, hashes


def faiss_index(centers, threads=1):
    import faiss
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatL2(centers.shape[1])
    index.add(np.ascontiguousarray(centers, dtype=np.float32))
    return index


def train_coarse(args):
    import faiss
    if min(args.dimension, args.nlist, args.sample_cap, args.batch_size, args.threads, args.iterations) <= 0:
        raise ValueError("positive dimensions/counts required")
    start = time.perf_counter()
    base, mapping, offsets, targets, cat, hashes = context(args.assets, args.catalog, args.dimension)
    training = context(args.training_assets, args.training_catalog, args.dimension) if args.training_assets else (
        base, mapping, offsets, targets, cat, hashes)
    tb, tm, to, tt, tc, th = training
    if getattr(args, "sample_ids", None):
        if args.training_assets:
            raise ValueError("shared base-graph sample cannot use a separate training graph")
        from scripts.edge_estimation.final_study.sample_edges import read_sample
        ids = read_sample(args.sample_ids, args.assets, args.catalog, args.dimension)
    else:
        ids = np.sort(np.random.default_rng(args.seed).choice(len(tt), min(args.sample_cap, len(tt)), replace=False))
    _, _, lengths, sample = edge_geometry(tb, tm, to, tt, ids)
    valid = lengths > 0
    if getattr(args, "sample_ids", None) and not valid.all():
        raise ValueError("shared sample contains invalid edges")
    ids, sample = ids[valid], sample[valid]
    if len(sample) < max(args.nlist, 256):
        raise ValueError("need at least max(nlist,256) nonzero training edges")
    faiss.omp_set_num_threads(args.threads)
    km = faiss.Kmeans(args.dimension, args.nlist, niter=args.iterations,
                      seed=args.seed, nredo=1, max_points_per_centroid=max(256, math.ceil(len(sample)/args.nlist)))
    validation_sampling_seconds = time.perf_counter()-start
    training_start = time.perf_counter()
    km.train(np.ascontiguousarray(sample))
    coarse_training_seconds = time.perf_counter()-training_start
    centers = np.asarray(km.centroids, dtype="<f4")
    index = faiss_index(centers, args.threads)
    with atomic_output_dir(args.out) as root:
        centers.tofile(root / "centers.f32le")
        sample.astype("<f4").tofile(root / "sample.f32le")
        ids.astype("<u8").tofile(root / "sample_ids.u64le")
        counts = np.zeros(args.nlist, dtype=np.int64)
        assignment_start = time.perf_counter()
        with (root / "assignments.u32le").open("wb") as stream:
            for first in range(0, len(targets), args.batch_size):
                eids = np.arange(first, min(first+args.batch_size, len(targets)))
                _, _, _, unit = edge_geometry(base, mapping, offsets, targets, eids)
                assigned = index.search(unit, 1)[1][:, 0].astype("<u4")
                assigned.tofile(stream)
                counts += np.bincount(assigned, minlength=args.nlist)
        assignment_seconds = time.perf_counter()-assignment_start
        residual = sample - centers[index.search(sample, 1)[1][:, 0]]
        manifest = {"format": "uq-ivf-edge-coarse/1", "dimension": args.dimension,
                    "nlist": args.nlist, "seed": args.seed, "edge_count": len(targets),
                    "catalog_identity": cat["catalog_identity"], "assets": hashes,
                    "training_assets": th, "training_catalog_identity": tc["catalog_identity"],
                    "training_scope": "separate_training_graph" if args.training_assets else "frozen_base_graph",
                    "sample_count": len(ids), "training_seconds": time.perf_counter()-start,
                    "validation_sampling_seconds": validation_sampling_seconds,
                    "coarse_training_seconds": coarse_training_seconds,
                    "assignment_seconds": assignment_seconds,
                    "sample_ids_sha256": sha256_file(root / "sample_ids.u64le"),
                    "faiss": faiss.__version__, "numpy": np.__version__,
                    "iterations": args.iterations, "threads": args.threads,
                    "list_statistics": {"mean": float(counts.mean()), "p95": float(np.quantile(counts, .95)),
                                        "max": int(counts.max()), "empty": int((counts == 0).sum())},
                    "residual_mean_squared_norm": float(np.mean(np.sum(residual.astype(float)**2, axis=1))),
                    "files": inventory(root)}
        dump(root / "manifest.json", manifest)
        dump(root / "complete.json", {"manifest_sha256": sha256_file(root / "manifest.json")})


def encode(args):
    start = time.perf_counter()
    root_coarse = Path(args.coarse)
    coarse = verify(root_coarse)
    d, nc = coarse["dimension"], coarse["nlist"]
    if args.m <= 0 or d % args.m or not 1 <= args.nbits <= 8 or args.batch_size <= 0:
        raise ValueError("invalid product codec shape")
    if args.qjl_bits <= 0 or args.qjl_bits % 8:
        raise ValueError("qjl_bits must be a positive multiple of 8")
    base, mapping, offsets, targets, cat, hashes = context(args.assets, args.catalog, d)
    if hashes != coarse["assets"] or cat["catalog_identity"] != coarse["catalog_identity"]:
        raise ValueError("coarse/base/index/mapping/catalog mismatch")
    centers = np.fromfile(root_coarse / "centers.f32le", dtype="<f4").reshape(nc, d)
    assignments = np.memmap(root_coarse / "assignments.u32le", dtype="<u4", mode="r")
    if len(assignments) != len(targets) or np.any(assignments >= nc):
        raise ValueError("invalid coarse assignment")
    sample = np.fromfile(root_coarse / "sample.f32le", dtype="<f4").reshape(-1, d)
    sample -= centers[faiss_index(centers, args.threads).search(sample, 1)[1][:, 0]]
    trainer = {"provider": "faiss", "seed": coarse["seed"], "iterations": args.iterations,
               "outer_iterations": args.opq_iterations, "threads": args.threads}
    if getattr(args, "trainer_config", None):
        from scripts.edge_estimation.contracts import load_strict_json
        trainer = load_strict_json(args.trainer_config)
        if trainer.get("provider") != "faiss" or trainer.get("seed") != coarse["seed"] or trainer.get("threads") != args.threads:
            raise ValueError("trainer config differs from coarse seed/provider/thread contract")
    validation_seconds = time.perf_counter()-start
    training_start = time.perf_counter()
    pq_identity = None
    if args.method == "pq_qjl":
        if not args.pq_artifact:
            raise ValueError("PQ+QJL requires --pq-artifact to share the exact PQ model")
        import faiss
        from scripts.edge_estimation.trainers.product_model import ProductCodebookModel
        pm = verify(args.pq_artifact)
        if (pm["method"] != "ivf_pq" or pm["coarse_identity"] != sha256_file(root_coarse / "manifest.json")
                or pm["m"] != args.m or pm["nbits"] != args.nbits or pm["assets"] != hashes):
            raise ValueError("PQ companion identity mismatch")
        pq = faiss.ProductQuantizer(d, args.m, args.nbits)
        books = np.fromfile(args.pq_artifact / "codebook.f32le", dtype="<f4").reshape(args.m, 1 << args.nbits, d//args.m)
        faiss.copy_array_to_vector(books.ravel(), pq.centroids)
        model = ProductCodebookModel("pq_packed", "shared_faiss_pq", "faiss", books, args.nbits,
            lambda x: np.asarray(faiss.unpack_bitstrings(pq.compute_codes(np.ascontiguousarray(x)), args.m, args.nbits), dtype=np.uint16),
            dependency_versions={"faiss": faiss.__version__, "numpy": np.__version__})
        pq_identity = sha256_file(args.pq_artifact / "manifest.json")
    else:
        model = train_product_model({"kind": "opq" if args.method == "opq" else "pq_packed",
                                     "m": args.m, "nbits": args.nbits}, trainer, sample)
    quantizer_training_seconds = time.perf_counter()-training_start
    rotation = model.rotation
    if rotation is not None and not np.allclose(rotation @ rotation.T, np.eye(d), atol=2e-4, rtol=2e-4):
        raise ValueError("OPQ rotation is not orthogonal")
    qjl = args.method == "pq_qjl"
    projection = np.random.default_rng(args.qjl_seed).standard_normal((args.qjl_bits, d)).astype("<f4") if qjl else None
    cb = (args.m * args.nbits + 7)//8
    stride = 20 + cb + (8 + args.qjl_bits//8 if qjl else 0)
    with atomic_output_dir(args.out) as root:
        centers.tofile(root / "centers.f32le")
        model.codebooks.astype("<f4").tofile(root / "codebook.f32le")
        if rotation is not None:
            rotation.astype("<f4").tofile(root / "rotation.f32le")
        if qjl:
            projection.tofile(root / "projection.f32le")
        encoding_start = time.perf_counter()
        with (root / "edges.bin").open("wb") as stream:
            for first in range(0, len(targets), args.batch_size):
                ids = np.arange(first, min(first+args.batch_size, len(targets)))
                _, source_rows, lengths, unit = edge_geometry(base, mapping, offsets, targets, ids)
                labels = np.asarray(assignments[ids])
                codes, _, recon = model.encode_batch(unit - centers[labels])
                # Decode in original coordinates in float64; the online rotation has identical orientation.
                h = centers[labels].astype(float) + (recon.astype(float) @ rotation.astype(float) if rotation is not None else recon.astype(float))
                source = base[source_rows].astype(float)
                delta = base[mapping[targets[ids]].astype(np.int64)].astype(float) - source
                offset = lengths**2 + 2*lengths*np.einsum("ij,ij->i", source, h)
                packed = pack_code_matrix(codes, args.nbits)
                payload = bytearray(len(ids)*stride)
                def field(dtype, offset, shape=()):
                    return np.ndarray((len(ids),)+shape, dtype=dtype, buffer=payload, offset=offset,
                                      strides=(stride,)+( (1,) if shape else ()))
                field("<f8", 0)[:] = lengths
                field("<u4", 16)[:] = labels
                field("u1", 20, (cb,))[:] = packed
                if qjl:
                    z = delta - lengths[:, None]*h
                    scale = np.linalg.norm(z, axis=1)*math.sqrt(math.pi/2)/args.qjl_bits
                    offset += 2*np.einsum("ij,ij->i", source, z)
                    signs = np.packbits(z @ projection.astype(float).T >= 0, axis=1, bitorder="little")
                    field("<f8", 20+cb)[:] = scale
                    field("u1", 28+cb, (args.qjl_bits//8,))[:] = signs
                field("<f8", 8)[:] = offset
                stream.write(payload)
        encoding_seconds = time.perf_counter()-encoding_start
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unavailable"
        manifest = {"format": "uq-ivf-edge/1", "method": "ivf_"+args.method,
                    "representation": "coarse_residual_unit_edge", "dimension": d, "nlist": nc,
                    "m": args.m, "nbits": args.nbits, "edge_count": len(targets), "record_size": stride,
                    "qjl_bits": args.qjl_bits if qjl else 0, "qjl_seed": args.qjl_seed if qjl else None,
                    "catalog_identity": cat["catalog_identity"], "assets": hashes,
                    "coarse_identity": sha256_file(root_coarse / "manifest.json"),
                    "coarse_manifest": coarse, "trainer": trainer, "pq_identity": pq_identity,
                    "validation_seconds": validation_seconds,
                    "quantizer_training_seconds": quantizer_training_seconds,
                    "encoding_seconds": encoding_seconds,
                    "dependency_versions": model.dependency_versions, "source_commit": commit,
                    "source_files": {str(p.relative_to(Path(__file__).resolve().parents[2])).replace("\\", "/"): sha256_file(p)
                        for p in [Path(__file__).resolve(), *sorted((Path(__file__).parent/"trainers").glob("*.py"))]},
                    "training_encoding_seconds": time.perf_counter()-start, "files": inventory(root)}
        dump(root / "manifest.json", manifest)
        config = {k: manifest[k] for k in ("format", "dimension", "nlist", "m", "nbits", "edge_count", "record_size", "qjl_bits", "catalog_identity", "coarse_identity")}
        config.update(backend=manifest["method"], coverage="full_graph", rotation=int(rotation is not None),
                      index_sha256=hashes["index"], manifest_sha256=sha256_file(root / "manifest.json"))
        for name, entry in manifest["files"].items():
            config[name+"_sha256"] = entry["sha256"]
        (root / "native.cfg").write_text("".join(f"{k}={v}\n" for k,v in config.items()), encoding="utf-8")
        dump(root / "complete.json", {"manifest_sha256": sha256_file(root / "manifest.json"),
                                      "native_sha256": sha256_file(root / "native.cfg")})
        # Fixed simple native-readable publication seal, kept separate from JSON parsing.
        (root / "complete.sha256").write_text(sha256_file(root / "native.cfg")+"\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("coarse")
    e = sub.add_parser("encode")
    for p in (c,e):
        p.add_argument("--assets", required=True, type=Path)
        p.add_argument("--catalog", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--batch-size", type=int, default=4096)
        p.add_argument("--iterations", type=int, default=25)
        p.add_argument("--threads", type=int, default=1)
    c.add_argument("--dimension", type=int, required=True)
    c.add_argument("--nlist", type=int, default=256)
    c.add_argument("--sample-cap", type=int, default=100000)
    c.add_argument("--sample-ids", type=Path)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--training-assets", type=Path)
    c.add_argument("--training-catalog", type=Path)
    e.add_argument("--coarse", type=Path, required=True)
    e.add_argument("--method", choices=("pq", "opq", "pq_qjl"), required=True)
    e.add_argument("--m", type=int, default=32)
    e.add_argument("--nbits", type=int, default=8)
    e.add_argument("--opq-iterations", type=int, default=25)
    e.add_argument("--trainer-config", type=Path)
    e.add_argument("--qjl-bits", type=int, default=128)
    e.add_argument("--qjl-seed", type=int, default=43)
    e.add_argument("--pq-artifact", type=Path)
    v = sub.add_parser("validate")
    v.add_argument("artifact", type=Path)
    args = parser.parse_args()
    if args.stage == "coarse":
        if bool(args.training_assets) != bool(args.training_catalog):
            parser.error("training-assets and training-catalog must be supplied together")
        train_coarse(args)
    elif args.stage == "encode":
        encode(args)
    else:
        print(json.dumps(verify(args.artifact), indent=2))


if __name__ == "__main__":
    main()
