"""Build or verify one graph, then train both methods from the same edge sample."""
from __future__ import annotations
import argparse,subprocess,sys,time
from pathlib import Path
import numpy as np
from .common import cli,write,load,unseal,seal,identity,verify_file
from .environment import ROOT,verify_build,runtime
from .schema import protocol
from .sample_edges import sample
from .asset_contract import bind
from scripts.edge_estimation.ivf_train import context

def prepare(a):
    p=protocol(a.protocol);data=unseal(a.dataset);build=verify_build(a.tools_build)
    training_environment=runtime(False)
    if build["profile"]!="tools" or not data.get("validated"):raise ValueError("validated data and tools build required")
    out=Path(a.out).resolve()
    if out.exists():raise ValueError("asset output exists; preserve it and use a new version")
    out.mkdir(parents=True);ledger={}
    def run(stage,cmd):
        start=time.perf_counter()
        with (out/(stage+".log")).open("w",encoding="utf-8") as log:
            subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,check=True)
        ledger[stage+"_wall_seconds"]=time.perf_counter()-start
    if bool(a.reuse_assets)!=bool(a.reuse_catalog):raise ValueError("reuse requires both assets and catalog")
    if a.reuse_assets:
        assets=load(a.reuse_assets);catalog=Path(a.reuse_catalog)
        context(a.reuse_assets,catalog,data["dimension"])
        # Historical build provenance is explicitly retained, never relabeled as a new one-thread graph.
        ledger["graph_status"]="reused_verified";ledger["graph_build_seconds"]=None
    else:
        labels=np.load(verify_file(data["base_labels"]),allow_pickle=False)
        if not np.array_equal(labels,np.arange(data["n_base"])):raise ValueError("new graph builder requires row-number labels; reuse a verified labeled graph otherwise")
        graphdir=out/"graph";graphdir.mkdir()
        dataset_config=Path(a.dataset).parent/"dataset.json"
        run("graph_build",[build["files"]["real_data_trace_runner"]["path"],
            "--dataset-config",dataset_config,"--mode","build-index","--index-path",graphdir/"index.bin",
            "--output-dir",graphdir,"--run-id",p["study_id"],"--M",16,"--ef-construction",200,"--seed",42,"--threads",1])
        np.save(out/"internal_to_base.npy",np.arange(data["n_base"],dtype="<u8"))
        assets={"base":data["base"]["path"],"index":str(graphdir/"index.bin"),
                "internal_to_label":str(out/"internal_to_base.npy")}
        catalog=out/"catalog"
        run("catalog",[build["files"]["uq_capture"]["path"],"--catalog","--index",assets["index"],
                       "--dimension",data["dimension"],catalog])
        ledger["graph_status"]="new_M16_efConstruction200_seed42_one_thread"
    write(out/"assets.json",assets)
    trainer={**p["training"]["trainer"]}
    started=time.perf_counter()
    shared=sample(out/"assets.json",catalog,data["dimension"],p["training"]["sample_count"],trainer["seed"],out/"sample")
    ledger["shared_sampling_wall_seconds"]=time.perf_counter()-started
    trainer["shared_sample_ids"]=str(shared)
    write(out/"trainer.json",trainer)
    config=load(ROOT/"configs/edge_estimation/methods/opq.json")
    config.update(run_id=p["study_id"],trainer=trainer)
    config["capture"]["dimension"]=data["dimension"]
    config["codec"].update(m=p["training"]["m"],nbits=p["training"]["nbits"])
    write(out/"opq_config.json",config)
    run("opq_train_encode",[sys.executable,"-m","scripts.edge_estimation.train_encode","--config",out/"opq_config.json",
        "--assets",out/"assets.json","--catalog",catalog,"--out",out/"opq"])
    shared_args=["--assets",out/"assets.json","--catalog",catalog,"--threads",1,"--batch-size",trainer["encode_batch_size"]]
    run("coarse",[sys.executable,"-m","scripts.edge_estimation.ivf_train","coarse",*shared_args,
        "--dimension",data["dimension"],"--nlist",p["training"]["coarse_centers"],"--seed",trainer["seed"],
        "--sample-ids",shared,"--iterations",p["training"]["coarse_iterations"],"--out",out/"coarse"])
    run("cr_opq_train_encode",[sys.executable,"-m","scripts.edge_estimation.ivf_train","encode",*shared_args,
        "--coarse",out/"coarse","--method","opq","--m",p["training"]["m"],"--nbits",p["training"]["nbits"],
        "--trainer-config",out/"trainer.json","--out",out/"ivf_opq"])
    bind(a.dataset,out/"assets.json",catalog,out/"opq",out/"ivf_opq",shared,out/"asset_contract.json")
    seal(out/"offline_costs.json",{"schema_version":2,"ledger":ledger,"protocol":identity(a.protocol),
        "training_environment":training_environment,
        "asset_contract":identity(out/"asset_contract.json"),"tools_build":identity(a.tools_build),
        "input_reuse": {"assets":identity(a.reuse_assets)} if a.reuse_assets else None})
    print(out/"asset_contract.json")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","dataset","tools-build","out"):p.add_argument("--"+key,required=True)
    p.add_argument("--reuse-assets");p.add_argument("--reuse-catalog");prepare(p.parse_args())
if __name__=="__main__":cli(main)
