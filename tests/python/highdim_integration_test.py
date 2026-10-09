"""Synthetic REAL-dimension Parquet -> GT -> graph/models -> search/freeze pipeline.

Run with a fresh final_study tools build. This is local smoke, never formal evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from highdim_data_test import fixture, convert, reviewed_controls
from scripts.edge_estimation.final_study.common import identity, load, unseal, write
from scripts.edge_estimation.final_study.environment import THREAD_VARS
from scripts.edge_estimation.final_study.query_contract import read_ids, write_ids
from scripts.datasets.prepare_highdim import profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--profile", choices=("bioasq1024", "dbpedia1536"), required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ.update({key: "1" for key in THREAD_VARS})
    spec = profile(args.profile)
    dataset = spec["dataset_id"]
    lock = fixture(root / "raw", name=args.profile, base_count=512, query_count=40)
    convert(lock, root / "converted", name=args.profile, batch=13,
            holdout=40 if args.profile == "dbpedia1536" else None)
    controls = reviewed_controls(root, root / "converted", counts=(10,10,20,2))
    commands = []
    def run(module, *arguments):
        cmd = [sys.executable, "-m", module, *map(str, arguments)]
        commands.append(cmd)
        log = root / f"step-{len(commands):02d}.log"
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise AssertionError(f"failed: {cmd}\n{log.read_text(encoding='utf-8')[-12000:]}")
    data = "scripts.datasets.final_study_data"
    prefix = "scripts.edge_estimation.final_study."
    run(data,"inventory","--registry",controls/"registry.highdim.json","--out",root/"inventory")
    run(data,"audit-queries","--registry",controls/"registry.highdim.json",
        "--history-roots",controls/"history.highdim.json","--out",root/"audit")
    run(data,"materialize","--dataset",dataset,"--audit",root/"audit/audit.json",
        "--split-spec",controls/"splits.highdim.json","--out",root/"data")
    for split in ("dev","select","test"):
        run(data,"exact-gt","--dataset-manifest",root/"data/dataset_manifest.json","--split",split,
            "--k",100,"--base-block",128,"--query-block",4,"--threads",1)
    run(data,"validate","--dataset-manifest",root/"data/dataset_manifest.json","--float64-check-queries",5)
    protocol = load(ROOT / "configs/edge_estimation/final_study/protocol.json")
    protocol["study_id"] = "synthetic-highdim-" + args.profile
    # Keep the sample above D: the installed Windows Faiss OPQ crashes for the
    # underdetermined 512x1024 fixture. Production uses the unchanged 100K budget.
    protocol["training"].update(m=32,nbits=4,coarse_centers=8,coarse_iterations=2,sample_count=2048)
    protocol["training"]["trainer"].update(iterations=2,outer_iterations=2,initial_pq_iterations=2,
                                          max_train_points=2048,max_points_per_centroid=256,encode_batch_size=128)
    protocol["development"].update(ef_grids={dataset:[40,80]},beta_grid=[1000000.0])
    protocol["selection"].update(recall_targets=[.90,.95],batch_recall_target=.95,noise_cv_limit=10,
                                  required_datasets=[dataset],mechanism_beta=1000000.0)
    protocol["measurement"]["minimum_case_seconds"] = .001
    write(root/"protocol.json",protocol)
    build = args.build.resolve()
    run(prefix+"prepare_assets","--protocol",root/"protocol.json","--dataset",root/"data/dataset_contract.json",
        "--tools-build",build/"build_manifest.json","--out",root/"assets")
    common = ["--protocol",root/"protocol.json","--dataset",root/"data/dataset_contract.json",
              "--assets",root/"assets/asset_contract.json","--build",build/"build_manifest.json"]
    def measure(cases,out,allocation=0):
        run(prefix+"run",*common,"--cases",cases,"--allocation",allocation,"--out",out,"--local-smoke")
    # Asset parity at batch=1/8/128 uses dev only, with a nonmultiple tail.
    build_doc = unseal(build/"build_manifest.json")
    doc = unseal(root/"data/dataset_contract.json")
    warm = set(read_ids(doc["warmup_ids"]["path"]))
    ids = [q for q in read_ids(doc["splits"]["dev"]["ids"]["path"]) if q not in warm][:7]
    write_ids(root/"dev-parity.ids",ids)
    capture = [build_doc["files"]["uq_capture"]["path"],"--index",str(root/"assets/graph/index.bin"),
               "--queries",doc["query_pool"]["path"],"--query-ids",str(root/"dev-parity.ids"),
               "--dimension",str(spec["dimension"]),"--k","10","--ef","40","--out",str(root/"trace")]
    with (root/"capture.log").open("w") as log:
        subprocess.run(capture,stdout=log,stderr=subprocess.STDOUT,check=True)
    for method in ("opq","ivf_opq"):
        for batch in (1,8,128):
            cmd=[build_doc["files"]["uq_native_runner"]["path"],"validate-batch-artifact",str(root/"assets"/method),
                 str(root/"trace/events.bin"),doc["query_pool"]["path"],str(batch),"1e-4","1e-5"]
            result = subprocess.run(cmd,text=True,capture_output=True,check=True)
            value=json.loads(result.stdout)
            assert value["valid"] and value["compared_count"]>0 and value["tolerance_failure_count"]==0
            assert value["nonfinite_mismatch_count"]==0
            write(root/f"parity-{method}-{batch}.json",value)
    run(prefix+"make_cases",*common,"--phase","dev","--out",root/"dev-cases.json")
    measure(root/"dev-cases.json",root/"dev")
    run(prefix+"make_cases",*common,"--phase","batch","--dev-runs",root/"dev","--out",root/"batch-cases.json")
    measure(root/"batch-cases.json",root/"batch")
    run(prefix+"select","--runs",root/"batch","--out",root/"batch-choice.json")
    run(prefix+"make_cases",*common,"--phase","select","--dev-runs",root/"dev",
        "--batch-choice",root/"batch-choice.json","--out",root/"select-cases.json")
    measure(root/"select-cases.json",root/"select")
    run(prefix+"recommend","--runs",root/"select","--bootstrap-draws",100,"--out",root/"recommendation.json")
    run(prefix+"freeze","--runs",root/"select","--method-selection",root/"recommendation.json",
        "--dev-runs",root/"dev","--out",root/"freeze.json","--diagnostic-out",root/"diagnostic-cases.json",
        "--mechanism-out",root/"mechanism-cases.json")
    tests=[]
    for allocation in range(3):
        dest=root/f"test-{allocation}";measure(root/"freeze.json",dest,allocation);tests.append(dest)
    measure(root/"diagnostic-cases.json",root/"diagnostic")
    measure(root/"mechanism-cases.json",root/"mechanism")
    run(prefix+"analyze","--freeze",root/"freeze.json","--runs",*tests,"--diagnostics",root/"diagnostic",
        "--mechanisms",root/"mechanism","--bootstrap-draws",100,"--out",root/"analysis.json")
    run(prefix+"cost","--assets",root/"assets/asset_contract.json","--analysis",root/"analysis.json",
        "--ledger",root/"assets/offline_costs.json","--out",root/"cost.json")
    analysis=unseal(root/"analysis.json")
    assert analysis["local_smoke"] and all(t["status"]=="OK" for t in analysis["targets"])
    write(root/"test-report.json",{"passed":True,"synthetic_only":True,"profile":args.profile,
          "dimension":spec["dimension"],"build":identity(build/"build_manifest.json"),"commands":commands,
          "checks":["Parquet conversion","audited splits","exhaustive GT/float64 check","graph/OPQ/CR training",
                    "six asset parities","dev/batch/select/recommend/freeze","test/diagnostic/mechanism/cost"]})
    print(root/"test-report.json")


if __name__ == "__main__":
    main()
