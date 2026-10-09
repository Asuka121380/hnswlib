"""Real train -> encode -> native replay -> HNSW test (requires Faiss)."""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.edge_estimation.contracts import read_events, sha256_file
from scripts.edge_estimation.ivf_train import verify
from scripts.edge_estimation.train_encode import read_fvecs


def run(command, ok=True):
    value = subprocess.run(list(map(str, command)), cwd=ROOT, text=True, capture_output=True)
    if (value.returncode == 0) != ok:
        raise AssertionError(f"{command}\n{value.stdout}\n{value.stderr}")
    return value.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    build = args.build.resolve()
    suffix = ".exe" if os.name == "nt" else ""
    exe = lambda name: build / (name+suffix)
    root = args.out.resolve() if args.out else Path(tempfile.mkdtemp(prefix="uq-ivf-"))
    root.mkdir(parents=True, exist_ok=True)
    fixture = root / "fixture"; fixture.mkdir()
    run([exe("uq_capture_fixture"), fixture])
    run([exe("uq_capture"), "--catalog", "--index", fixture / "index.bin", "--dimension", 8, root / "catalog"])
    run([exe("uq_capture"), "--index", fixture / "index.bin", "--queries", fixture / "queries.fvecs",
         "--query-ids", fixture / "query_ids.txt", "--dimension", 8, "--k", 10, "--ef", 40, "--out", root / "trace"])
    # Nonidentity base-row mapping without changing the persisted graph or query data.
    values = np.array(read_fvecs(fixture / "base.fvecs", 8))
    perm = np.random.default_rng(90).permutation(len(values))
    mapping = np.argsort(perm).astype("<u8")
    np.save(root / "mapping.npy", mapping)
    with (root / "base.fvecs").open("wb") as out:
        for row in values[perm]:
            np.array([8], dtype="<i4").tofile(out); row.astype("<f4").tofile(out)
    assets = {"base": str(root / "base.fvecs"), "index": str(fixture / "index.bin"),
              "internal_to_label": str(root / "mapping.npy")}
    (root / "assets.json").write_text(json.dumps(assets))
    script = ROOT / "scripts/edge_estimation/ivf_train.py"
    common = ["--assets", root / "assets.json", "--catalog", root / "catalog", "--batch-size", 64, "--iterations", 3]
    run([sys.executable, script, "coarse", *common, "--dimension", 8, "--nlist", 8,
         "--sample-cap", 512, "--out", root / "coarse"])
    queries = np.array(read_fvecs(fixture / "queries.fvecs", 8), dtype=float)
    _, events = read_events(root / "trace/events.bin")
    report = {}
    baseline_results = None
    for method in ("pq", "opq", "pq_qjl"):
        artifact = root / method
        command = [sys.executable, script, "encode", *common, "--coarse", root / "coarse", "--method", method,
                   "--m", 2, "--nbits", 4, "--opq-iterations", 2, "--qjl-bits", 32, "--out", artifact]
        if method == "pq_qjl": command += ["--pq-artifact", root / "pq"]
        run(command)
        manifest = verify(artifact)
        cb = np.fromfile(artifact / "codebook.f32le", dtype="<f4").reshape(2,16,4).astype(float)
        centers = np.fromfile(artifact / "centers.f32le", dtype="<f4").reshape(8,8).astype(float)
        rotation = np.fromfile(artifact / "rotation.f32le", dtype="<f4").reshape(8,8).astype(float) if method == "opq" else np.eye(8)
        if method == "opq": assert not np.allclose(rotation, np.eye(8))
        projection = np.fromfile(artifact / "projection.f32le", dtype="<f4").reshape(32,8).astype(float) if method == "pq_qjl" else None
        records = (artifact / "edges.bin").read_bytes()
        scores = list(csv.DictReader(io.StringIO(run([exe("uq_native_runner"), "scores-artifact", artifact,
                                            root / "trace/events.bin", fixture / "queries.fvecs"]))))
        lookup = {int(row["event_id"]): float(row["estimate"]) for row in scores}
        errors = []
        for event in events:
            if event.kind != 3: continue
            start = event.edge_id*manifest["record_size"]
            record = records[start:start+manifest["record_size"]]
            length, offset = np.frombuffer(record[:16], dtype="<f8")
            center = int(np.frombuffer(record[16:20], dtype="<u4")[0])
            codes = [record[20]&15, record[20]>>4]
            h = centers[center] + np.concatenate([cb[0,codes[0]], cb[1,codes[1]]]) @ rotation
            q, c, v = queries[event.query_id], values[event.source_id].astype(float), values[event.target_id].astype(float)
            # Independent geometry, not the exported offset, is the reference.
            expected = event.d_current + length*length - 2*length*np.dot(q-c,h)
            if projection is not None:
                z = v-c-length*h
                scale = np.linalg.norm(z)*np.sqrt(np.pi/2)/32
                signs = np.where(projection @ z >= 0, 1., -1.)
                stored = np.unpackbits(np.frombuffer(record[29:], dtype="u1"), bitorder="little")
                np.testing.assert_array_equal(stored, signs>0)
                expected += -2*scale*np.dot(projection @ q, signs) + 2*np.dot(c,z)
            observed = lookup[event.event_id]
            np.testing.assert_allclose(observed, expected, atol=1e-4, rtol=1e-5)
            errors.append(abs(observed-expected))
        parity = json.loads(run([exe("uq_native_runner"), "validate-batch-artifact", artifact,
                                root / "trace/events.bin", fixture / "queries.fvecs", 2]))
        assert parity["valid"]
        quality = json.loads(run([exe("uq_native_runner"), "quality", root / "trace/events.bin",
                                 root / "trace/labels.bin", artifact, fixture / "queries.fvecs", 1.0]))
        perf_common = [exe("uq_ivf_performance_runner"), "--index-path", fixture / "index.bin", "--artifact-path", artifact,
                       "--query-path", fixture / "queries.fvecs", "--dimension", 8, "--query-count", 6,
                       "--ef-search", 96, "--warmup-queries", 2, "--batch-size", 2]
        run([*perf_common, "--beta", 1e12, "--output", root / f"{method}.json",
             "--result-records-output", root / f"{method}.csv"])
        metrics = json.loads((root / f"{method}.json").read_text())
        assert metrics["attempted_estimates"] > 0 and metrics["pruned_estimates"] == 0
        run([*perf_common, "--no-prune", "--output", root / "baseline.json", "--result-records-output", root / "baseline.csv"])
        assert (root / f"{method}.csv").read_bytes() == (root / "baseline.csv").read_bytes()
        for row in csv.DictReader((root / "baseline.csv").open()):
            qid, rank = int(row["query_id"]), int(row["rank"])
            exact_ids = np.argsort(np.sum((values.astype(float)-queries[qid])**2,axis=1))[:10]+1000
            assert int(row["label"]) == exact_ids[rank]
        pruned_common = list(perf_common)
        pruned_common[pruned_common.index("--ef-search")+1] = 20
        run([*pruned_common, "--beta", .2, "--output", root / f"{method}-pruned.json"])
        assert json.loads((root / f"{method}-pruned.json").read_text())["pruned_estimates"] > 0
        report[method] = {"oracle_count": len(errors), "max_absolute_error": max(errors),
                          "batch_parity": parity, "quality": quality, "no_prune_topk_exact": True}
    # Shared primary PQ bytes are exact, not merely equal hyperparameters.
    pq = verify(root / "pq"); qjl = verify(root / "pq_qjl")
    a=(root / "pq/edges.bin").read_bytes(); b=(root / "pq_qjl/edges.bin").read_bytes()
    for i in range(pq["edge_count"]):
        assert a[i*pq["record_size"]+16:i*pq["record_size"]+21] == b[i*qjl["record_size"]+16:i*qjl["record_size"]+21]
    corrupt=root / "corrupt"; shutil.copytree(root / "pq", corrupt)
    with (corrupt / "edges.bin").open("r+b") as file: file.truncate(7)
    run([exe("uq_native_runner"), "validate-artifact", corrupt, root / "trace/events.bin", fixture / "queries.fvecs"], ok=False)
    report["truncation_rejected"] = True
    bad_config=root/"bad-config"; shutil.copytree(root/"pq",bad_config)
    cfgfile=bad_config/"native.cfg"
    original=cfgfile.read_text()
    cfgfile.write_text(original.replace("nbits=4", "nbits=8"))
    run([exe("uq_native_runner"),"validate-artifact",bad_config,root/"trace/events.bin",fixture/"queries.fvecs"],ok=False)
    cfgfile.write_text(original.replace(pq["catalog_identity"],"0"*64))
    (bad_config/"complete.sha256").write_text(sha256_file(cfgfile)+"\n")
    run([exe("uq_native_runner"),"validate-artifact",bad_config,root/"trace/events.bin",fixture/"queries.fvecs"],ok=False)
    report["config_and_wrong_catalog_rejected"]=True
    bad_mapping=dict(assets)
    np.save(root/"wrong-mapping.npy",np.arange(len(values),dtype="<u8"))
    bad_mapping["internal_to_label"]=str(root/"wrong-mapping.npy")
    (root/"wrong-assets.json").write_text(json.dumps(bad_mapping))
    run([sys.executable,script,"encode","--assets",root/"wrong-assets.json","--catalog",root/"catalog",
         "--coarse",root/"coarse","--method","pq","--m",2,"--nbits",4,"--out",root/"must-not-exist"],ok=False)
    assert not (root/"must-not-exist").exists()
    report["changed_mapping_rejected"]=True
    experiment_config={"architecture":"hnsw_coarse_residual_edges","exploratory":True,
        "dimension":8,"m":2,"nbits":4,"query_start":0,"query_count":6,"k":10,"warmup_queries":2,
        "repeats":1,"batch_size":2,"batch_grid":[1,2],"sample_cap":512,"coarse_centers":[8],
        "iterations":3,"opq_iterations":2,"qjl_bits":32,"encode_batch_size":64,
        "methods":{"hnsw":{"ef_search":[40]},**{f"ivf_{m}":{"ef_search":[40],"beta":[1.2]} for m in ("pq","opq","pq_qjl")}}}
    dump_path=root/"experiment.json"; dump_path.write_text(json.dumps(experiment_config))
    truth=np.argsort(np.sum((queries[:,None,:]-values.astype(float)[None,:,:])**2,axis=2),axis=1)[:,:10]+1000
    with (root/"truth.ivecs").open("wb") as out:
        for row in truth:
            np.array([10],dtype="<i4").tofile(out); row.astype("<i4").tofile(out)
    experiments=ROOT/"scripts/edge_estimation/ivf_experiments.py"
    run([sys.executable,experiments,"train","--config",dump_path,"--assets",root/"assets.json",
         "--catalog",root/"catalog","--out",root/"pipeline"])
    run([sys.executable,experiments,"quality","--config",dump_path,"--artifacts",root/"pipeline/k8",
         "--trace",root/"trace","--queries",fixture/"queries.fvecs","--native",exe("uq_native_runner"),"--out",root/"quality"])
    run([sys.executable,experiments,"grid","--config",dump_path,"--artifacts",root/"pipeline/k8",
         "--index",fixture/"index.bin","--queries",fixture/"queries.fvecs","--ground-truth",root/"truth.ivecs",
         "--runner",exe("uq_ivf_performance_runner"),"--out",root/"grid"])
    assert json.loads((root/"grid/complete.json").read_text())["points"]==7
    run([sys.executable,experiments,"refine","--config",dump_path,"--points",root/"grid","--out",root/"followups"])
    assert json.loads((root/"followups/confirm.json").read_text())["repeats"]==3
    experiment_config["exploratory"]=False
    dump_path.write_text(json.dumps(experiment_config))
    run([sys.executable,experiments,"quality","--config",dump_path,"--artifacts",root/"pipeline/k8",
         "--trace",root/"trace","--queries",fixture/"queries.fvecs","--native",exe("uq_native_runner"),
         "--out",root/"overlap-rejected"],ok=False)
    report["calibration_overlap_rejected"]=True
    report["experiment_pipeline"]={"quality":True,"grid_points":7,"summary":True}
    from uq_ivf_dimension_oracle import dimension_oracle
    report["dimension_960_oracle"]=dimension_oracle(root/"dimension960",exe("uq_native_runner"),run)
    (root / "test_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"passed": True, "report": str(root / "test_report.json"),
                      "max_errors": {m:report[m]["max_absolute_error"] for m in ("pq","opq","pq_qjl")}}))


if __name__ == "__main__": main()
