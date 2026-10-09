"""Generate bounded development, batch and select sweeps without reading test results."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import cli,identity,seal,unseal
from .schema import protocol,DEFAULT_EF,unique_cases,METHODS
from .results import rows,aggregate

def binding(a):
    return {key:identity(getattr(a,key)) for key in ("protocol","dataset","assets","build")}
def c(method,ef,beta=None,window=128,chunk=128):
    return {"method":method,"ef_search":ef,"beta":beta,"prepare_window":window,"compute_chunk":chunk,"inner_repeats":1}
def candidates(points,method,target):
    pool=[x for x in points if x["case"]["method"]==method]
    qualified=[x for x in pool if x["recall_at_k"]>=target]
    if not pool:raise ValueError("missing development method "+method)
    anchor=max(qualified,key=lambda x:x["qps_service"]) if qualified else max(pool,key=lambda x:x["recall_at_k"])
    config=anchor["case"]
    upper=sorted([x for x in pool if x["case"]["beta"]==config["beta"] and x["case"]["ef_search"]>config["ef_search"]],
                 key=lambda x:x["case"]["ef_search"])
    return [config]+([upper[0]["case"]] if upper else [])

def generate(a):
    p=protocol(a.protocol);data=unseal(a.dataset);bindings=binding(a);values=[];evidence=[];smoke=False
    if a.phase=="dev":
        grid=p.get("development",{}).get("ef_grids",{}).get(data["dataset_id"],DEFAULT_EF.get(data["dataset_id"]))
        if not grid:raise ValueError("dataset has no declared development ef grid")
        for method in METHODS:
            for ef in grid:
                for beta in ([None] if method=="hnsw" else p["development"]["beta_grid"]):
                    values.append(c(method,ef,beta))
    else:
        raw,manifests=rows(a.dev_runs,"dev",bindings);points=aggregate(raw)
        smoke=any(m["local_smoke"] for m in manifests)
        evidence=[identity(Path(x)/"complete.json" if Path(x).is_dir() else x) for x in a.dev_runs]
        if a.phase=="batch":
            for method in METHODS:
                anchor=candidates(points,method,p["selection"]["batch_recall_target"])[0]
                for window,chunk in ([(128,128)] if method=="hnsw" else [(128,128),("all",128),("all","all")]):
                    values.append({**anchor,"prepare_window":window,"compute_chunk":chunk})
        else:
            chosen=unseal(a.batch_choice)
            smoke=smoke or chosen["local_smoke"]
            if chosen["bindings"]!=bindings:raise ValueError("batch choice uses different inputs")
            evidence.append(identity(a.batch_choice));window,chunk=chosen["prepare_window"],chosen["compute_chunk"]
            for method in METHODS:
                for target in p["selection"]["recall_targets"]:
                    for config in candidates(points,method,target):
                        values.append({**config,"prepare_window":window,"compute_chunk":chunk})
    if Path(a.out).exists():raise ValueError("case files are immutable; choose a new output")
    seal(a.out,{"schema_version":2,"phase":a.phase,"dataset_id":data["dataset_id"],
        "bindings":bindings,"cases":unique_cases(values,p["search"]["k"]),"evidence":evidence,"local_smoke":smoke})
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","dataset","assets","build","out"):p.add_argument("--"+key,required=True)
    p.add_argument("--phase",choices=("dev","batch","select"),required=True)
    p.add_argument("--dev-runs",nargs="*",default=[]);p.add_argument("--batch-choice")
    generate(p.parse_args())
if __name__=="__main__":cli(main)
