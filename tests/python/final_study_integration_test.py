"""Real Faiss, native graph and full select/freeze/test study integration on synthetic data."""
from __future__ import annotations
import argparse,csv,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.edge_estimation.final_study.common import write,load,unseal,seal,identity
from scripts.edge_estimation.final_study.query_contract import write_vecs,evaluate_results,read_ids
from scripts.edge_estimation.final_study.environment import THREAD_VARS

def main():
    p=argparse.ArgumentParser();p.add_argument("--build",required=True);p.add_argument("--out")
    a=p.parse_args();root=Path(a.out).resolve() if a.out else Path(tempfile.mkdtemp(prefix="final-study-"))
    root.mkdir(parents=True,exist_ok=True)
    os.environ.update({key:"1" for key in THREAD_VARS})
    def run(module,*args,ok=True):
        result=subprocess.run([sys.executable,"-m",module,*map(str,args)],cwd=ROOT,text=True,capture_output=True)
        if (result.returncode==0)!=ok:raise AssertionError(f"{module} {args}\n{result.stdout}\n{result.stderr}")
        return result.stdout
    prefix="scripts.edge_estimation.final_study."
    data_cli="scripts.datasets.final_study_data"
    rng=np.random.default_rng(871)
    write_vecs(root/"base.fvecs",rng.normal(size=(512,8)))
    write_vecs(root/"query.fvecs",rng.normal(size=(40,8)))
    write(root/"registry.json",{"datasets":{"toy":{"dimension":8,"n_base":512,"metric":"squared_l2",
          "base":{"path":str(root/"base.fvecs")},"query_sources":{"new":{"path":str(root/"query.fvecs")}}}}})
    (root/"review.txt").write_text("Synthetic source generated during this test; no earlier runs exist.",encoding="utf-8")
    write(root/"review.json",[{"dataset_id":"toy","source":"new","source_sha256":identity(root/"query.fvecs")["sha256"],
         "status":"unused_confirmed","range":[0,40],"evidence":str(root/"review.txt"),"review_complete":True}])
    write(root/"history.json",{"roots":[],"review_ledger":str(root/"review.json")})
    write(root/"splits.json",{"datasets":{"toy":{"dev":{"source":"new","count":10,"warmup_count":2},
         "select":{"source":"new","count":10},"test":{"source":"new","count":20}}}})
    run(data_cli,"inventory","--registry",root/"registry.json","--out",root/"inventory")
    run(data_cli,"audit-queries","--registry",root/"registry.json","--history-roots",root/"history.json","--out",root/"audit")
    run(data_cli,"materialize","--dataset","toy","--audit",root/"audit/audit.json",
        "--split-spec",root/"splits.json","--out",root/"dataset")
    # Prove that absence of history does not imply fresh.
    write(root/"unreviewed.json",{"roots":[]})
    run(data_cli,"audit-queries","--registry",root/"registry.json","--history-roots",root/"unreviewed.json","--out",root/"unreviewed")
    run(data_cli,"materialize","--dataset","toy","--audit",root/"unreviewed/audit.json",
        "--split-spec",root/"splits.json","--out",root/"must-not-publish",ok=False)
    for split in ("dev","select","test"):
        run(data_cli,"exact-gt","--dataset-manifest",root/"dataset/dataset_manifest.json","--split",split,
            "--k",100,"--threads",1,"--base-block",128,"--query-block",4)
    run(data_cli,"validate","--dataset-manifest",root/"dataset/dataset_manifest.json","--float64-check-queries",5)
    protocol=load(ROOT/"configs/edge_estimation/final_study/protocol.json")
    protocol["study_id"]="synthetic-integration"
    protocol["training"].update(m=2,nbits=4,coarse_centers=8,coarse_iterations=2,sample_count=512)
    protocol["training"]["trainer"].update(iterations=2,outer_iterations=2,initial_pq_iterations=2,
        max_train_points=512,max_points_per_centroid=256,encode_batch_size=128)
    protocol["development"].update(ef_grids={"toy":[40,80]},beta_grid=[1000000.0])
    protocol["selection"].update(recall_targets=[.90,.95],batch_recall_target=.95,noise_cv_limit=10)
    protocol["selection"].update(required_datasets=["toy"],mechanism_beta=1000000.0)
    protocol["measurement"].update(minimum_case_seconds=.001)
    write(root/"protocol.json",protocol)
    build=Path(a.build).resolve()
    run(prefix+"prepare_assets","--protocol",root/"protocol.json","--dataset",root/"dataset/dataset_contract.json",
        "--tools-build",build/"build_manifest.json","--out",root/"assets")
    common=["--protocol",root/"protocol.json","--dataset",root/"dataset/dataset_contract.json",
            "--assets",root/"assets/asset_contract.json","--build",build/"build_manifest.json"]
    def measure(cases,out,allocation=0):
        return run(prefix+"run",*common,"--cases",cases,"--allocation",allocation,"--out",out,"--local-smoke")
    run(prefix+"make_cases",*common,"--phase","dev","--out",root/"dev_cases.json")
    measure(root/"dev_cases.json",root/"dev")
    run(prefix+"make_cases",*common,"--phase","batch","--dev-runs",root/"dev","--out",root/"batch_cases.json")
    measure(root/"batch_cases.json",root/"batch")
    run(prefix+"select","--runs",root/"batch","--out",root/"batch_choice.json")
    run(prefix+"make_cases",*common,"--phase","select","--dev-runs",root/"dev",
        "--batch-choice",root/"batch_choice.json","--out",root/"select_cases.json")
    measure(root/"select_cases.json",root/"select")
    run(prefix+"recommend","--runs",root/"select","--bootstrap-draws",100,"--out",root/"recommendation.json")
    run(prefix+"freeze","--runs",root/"select","--out",root/"freeze.json","--diagnostic-out",root/"diagnostic.json",
        "--method-selection",root/"recommendation.json","--dev-runs",root/"dev","--mechanism-out",root/"mechanism.json")
    test_runs=[]
    for allocation in range(3):
        dest=root/f"test-{allocation}";measure(root/"freeze.json",dest,allocation);test_runs.append(dest)
    # Resume must validate hashes without changing completed measurements.
    before=identity(test_runs[0]/"block-00"/unseal(root/"freeze.json")["cases"][0]["case_config_id"]/"performance.json")
    measure(root/"freeze.json",test_runs[0])
    assert identity(before["path"])==before
    measure(root/"diagnostic.json",root/"diagnostic")
    measure(root/"mechanism.json",root/"mechanism")
    run(prefix+"analyze","--freeze",root/"freeze.json","--runs",*test_runs,
        "--diagnostics",root/"diagnostic","--mechanisms",root/"mechanism","--bootstrap-draws",100,"--out",root/"analysis.json")
    run(prefix+"cost","--assets",root/"assets/asset_contract.json","--analysis",root/"analysis.json",
        "--ledger",root/"assets/offline_costs.json","--out",root/"cost.json")
    analysis=unseal(root/"analysis.json")
    assert analysis["local_smoke"] and all(t["status"]=="OK" for t in analysis["targets"])
    assert all(d["exact_l2_calls_per_query"]>0 for d in analysis["diagnostics"])
    assert analysis["mechanisms"] and len({c["inner_repeats"] for c in unseal(root/"freeze.json")["cases"]})==1
    assert all(d["case"]["inner_repeats"]==1 for d in analysis["diagnostics"])
    # Exercise noncontiguous order, inner repeats, tail windows, baseline-without-artifacts.
    build_doc=unseal(build/"build_manifest.json")
    binary=build_doc["files"]["uq_opq_performance_runner"]["path"]
    dataset=unseal(root/"dataset/dataset_contract.json")
    ids=read_ids(dataset["splits"]["test"]["ids"]["path"])
    (root/"noncontiguous.ids").write_text("\n".join(map(str,[ids[9],ids[0],ids[5]]))+"\n")
    cmd=[binary,"--no-prune","--index-path",str(root/"assets/graph/index.bin"),"--query-path",dataset["query_pool"]["path"],
         "--dimension","8","--query-ids",str(root/"noncontiguous.ids"),"--warmup-query-ids",dataset["warmup_ids"]["path"],
         "--prepare-window","2","--compute-chunk","2","--repeats","2","--metrics-level","diagnostic",
         "--output",str(root/"tail.json"),"--result-records-output",str(root/"tail.csv")]
    subprocess.run(cmd,check=True,capture_output=True)
    result=list(csv.DictReader((root/"tail.csv").open()))
    assert len(result)==60 and {int(r["repeat_id"]) for r in result}=={0,1}
    for method in ("opq","ivf_opq"):
        for window,chunk in ((2,2),("all",2),("all","all")):
            pruned=list(cmd);pruned.remove("--no-prune")
            pruned[0]=build_doc["files"]["uq_ivf_performance_runner" if method=="ivf_opq" else "uq_opq_performance_runner"]["path"]
            pruned[pruned.index("--prepare-window")+1]=str(window)
            pruned[pruned.index("--compute-chunk")+1]=str(chunk)
            pruned[pruned.index("--output")+1]=str(root/"batch-parity.json")
            pruned[pruned.index("--result-records-output")+1]=str(root/"batch-parity.csv")
            pruned += ["--artifact-path",str(root/"assets"/method),"--beta","1000000"]
            subprocess.run(pruned,check=True,capture_output=True)
            assert (root/"tail.csv").read_bytes()==(root/"batch-parity.csv").read_bytes()
    (root/"duplicate.ids").write_text("1\n1\n")
    bad=list(cmd);bad[bad.index("--query-ids")+1]=str(root/"duplicate.ids")
    assert subprocess.run(bad,capture_output=True).returncode!=0
    # Corrupted completed output must never be resumed as success.
    Path(before["path"]).write_text("{}")
    run(prefix+"run",*common,"--cases",root/"freeze.json","--out",test_runs[0],"--local-smoke",ok=False)
    write(root/"test_report.json",{"passed":True,"root":str(root),"synthetic_only":True,
        "checks":["freshness review required","exact GT float64 validation","same edge sample/budgets","shared graph and labels",
          "three batch strategies","dev/select/test isolation","immutable freeze","three paired allocations",
          "actual L2 diagnostics","cost ledger","resume identities","corrupt-output rejection","noncontiguous IDs and repeats"]})
    print(root/"test_report.json")
if __name__=="__main__":main()
