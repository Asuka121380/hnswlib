"""Choose a global preferred method from all preregistered select datasets, never test."""
from __future__ import annotations
import argparse,math
from pathlib import Path
import numpy as np
from .common import cli,seal,identity,seed
from .schema import protocol
from .results import rows,aggregate

def recommend(a):
    grouped={}
    # Read datasets separately: allocation/block IDs are intentionally local to each dataset.
    for path in a.runs:
        raw,manifests=rows([path],"select")
        key=manifests[0]["dataset_id"];grouped.setdefault(key,[]).append(path)
    all_raw={};all_manifests=[]
    for dataset,paths in grouped.items():
        all_raw[dataset],manifests=rows(paths,"select");all_manifests+=manifests
    binding=all_manifests[0]["bindings"];p=protocol(binding["protocol"]["path"])
    if any(m["bindings"]["protocol"]!=binding["protocol"] or m["bindings"]["build"]!=binding["build"] for m in all_manifests):
        raise ValueError("global method choice needs the same protocol/build")
    if not set(p["selection"]["required_datasets"]).issubset(grouped):raise ValueError("missing required select dataset")
    smoke=any(m["local_smoke"] for m in all_manifests);comparisons=[];paired={}
    for dataset,raw in sorted(all_raw.items()):
        points=aggregate(raw);paired[dataset]=[]
        if any(x["count"]<3 for x in points) and not smoke:raise ValueError("select evidence needs at least three blocks")
        if any(x["qps_cv"]>p["selection"]["noise_cv_limit"] and x["count"]<9 for x in points):
            raise ValueError("resolve noisy select measurements before global method selection")
        for target in p["selection"]["recall_targets"]:
            selected={}
            for method in ("hnsw","opq","ivf_opq"):
                candidates=[x for x in points if x["case"]["method"]==method and x["recall_at_k"]>=target
                            and x["scratch_bytes_max"]<=p["selection"]["scratch_budget_bytes"]]
                if not candidates:raise ValueError("no quality/memory-qualified select candidate")
                selected[method]=max(candidates,key=lambda x:x["qps_service"])
            block_values={}
            for method,point in selected.items():
                block_values[method]={(r["allocation_id"],r["block_id"]):r["qps_service"] for r in raw if r["case_config_id"]==point["case_config_id"]}
            keys=set(block_values["hnsw"])
            if any(set(v)!=keys for v in block_values.values()):raise ValueError("unpaired select blocks")
            ratios={key:block_values["ivf_opq"][key]/block_values["opq"][key] for key in keys}
            paired[dataset].append(ratios)
            comparisons.append({"dataset_id":dataset,"target":target,
               "cr_over_opq":math.exp(float(np.mean(np.log(list(ratios.values()))))),
               "selected_cases":{method:point["case_config_id"] for method,point in selected.items()}})
    rng=np.random.default_rng(seed(p["study_id"],"global-method"));boot=[]
    # Datasets/targets have fixed equal weight. Shared blocks are resampled jointly
    # across targets so reused cases never become independent observations.
    for _ in range(a.bootstrap_draws):
        values=[]
        for dataset,targets in paired.items():
            keys=sorted(targets[0])
            if any(set(t)!=set(keys) for t in targets):raise ValueError("target block coverage differs")
            chosen=rng.integers(0,len(keys),len(keys))
            values.extend(float(np.mean([math.log(t[keys[i]]) for i in chosen])) for t in targets)
        boot.append(math.exp(float(np.mean(values))))
    point=math.exp(float(np.mean([math.log(c["cr_over_opq"]) for c in comparisons])))
    low,high=map(float,np.quantile(boot,[.025,.975]))
    minimum=min(c["cr_over_opq"] for c in comparisons);maximum=max(c["cr_over_opq"] for c in comparisons)
    preferred="no_stable_global_winner"
    if point>1.02 and low>1 and minimum>=.98:preferred="ivf_opq"
    elif point<1/1.02 and high<1 and maximum<=1/.98:preferred="opq"
    if Path(a.out).exists():raise ValueError("method selection is immutable")
    seal(a.out,{"schema_version":2,"protocol":binding["protocol"],"build":binding["build"],
         "datasets":sorted(grouped),"local_smoke":smoke,"recommended_method":preferred,
         "equal_weight_cr_over_opq":point,"ci95":[low,high],"worst_cr_over_opq":minimum,
         "comparisons":comparisons,"evidence":[identity(Path(x)/"complete.json" if Path(x).is_dir() else x) for x in a.runs],
         "rule":"equal weight per dataset/target; paired select blocks; >2% gain, CI excludes 1, no cell regression >2%"})
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--runs",nargs="+",required=True)
    p.add_argument("--out",required=True);p.add_argument("--bootstrap-draws",type=int,default=10000)
    a=p.parse_args()
    if a.bootstrap_draws<100:raise ValueError("need at least 100 draws")
    recommend(a)
if __name__=="__main__":cli(main)
