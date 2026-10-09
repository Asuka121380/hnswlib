"""Freeze select-only anchors, common repetition budgets and dev-selected mechanisms."""
from __future__ import annotations
import argparse,math
from pathlib import Path
from .common import cli,seal,unseal,identity
from .schema import protocol,unique_cases,METHODS
from .results import rows,aggregate
from .query_contract import read_ids

def freeze(a):
    raw,manifests=rows(a.runs,"select");bindings=manifests[0]["bindings"]
    if any(m["bindings"]!=bindings for m in manifests):raise ValueError("mixed select inputs")
    p=protocol(bindings["protocol"]["path"]);data=unseal(bindings["dataset"]["path"])
    points=aggregate(raw);smoke=any(m["local_smoke"] for m in manifests)
    if any(x["count"]<3 for x in points) and not smoke:raise ValueError("need at least three selection blocks")
    if any(x["qps_cv"]>p["selection"]["noise_cv_limit"] and x["count"]<9 for x in points):
        raise ValueError("selection noise exceeds threshold; use nine blocks")
    strategies={(str(x["case"]["prepare_window"]),str(x["case"]["compute_chunk"])) for x in points}
    if len(strategies)!=1:raise ValueError("selection must use one common batching strategy")
    recommendation=None
    if a.method_selection:
        recommendation=unseal(a.method_selection)
        if recommendation["protocol"]!=bindings["protocol"] or data["dataset_id"] not in recommendation["datasets"]:
            raise ValueError("global method selection does not include this protocol/dataset")
        inputs={x["sha256"] for x in recommendation["evidence"]}
        if not all(identity(Path(x)/"complete.json" if Path(x).is_dir() else x)["sha256"] in inputs for x in a.runs):
            raise ValueError("global method selection is not based on these select runs")
        if recommendation["local_smoke"] and not smoke:raise ValueError("cannot use smoke method selection")
    elif not smoke:
        raise ValueError("formal freeze requires --method-selection from all preregistered datasets")
    selected=[];provisional=[]
    for method in METHODS:
        pool=[x for x in points if x["case"]["method"]==method and x["scratch_bytes_max"]<=p["selection"]["scratch_budget_bytes"]]
        for target in p["selection"]["recall_targets"]:
            qualified=[x for x in pool if x["recall_at_k"]>=target]
            if not qualified:raise ValueError(f"{method} did not reach {target} within the memory budget; refine dev and repeat select")
            anchor=max(qualified,key=lambda x:x["qps_service"]);cfg=anchor["case"]
            upper=sorted([x for x in pool if x["case"]["beta"]==cfg["beta"] and x["case"]["ef_search"]>cfg["ef_search"]],
                         key=lambda x:x["case"]["ef_search"])
            neighbors=[anchor]+upper[:1];selected.extend(neighbors)
            provisional.append((method,target,neighbors))
    nq=len(read_ids(data["splits"]["test"]["ids"]["path"]))
    # One R across the dataset makes all quality-target comparisons directly paired;
    # the fastest selected case sets the minimum wall-time budget.
    repeats=max(1,math.ceil(p["measurement"]["minimum_case_seconds"]*max(x["qps_service"] for x in selected)/nq))
    if repeats*nq*p["search"]["k"]*16>p["measurement"]["result_buffer_cap_bytes"]:
        raise ValueError("common repetition budget exceeds result buffer cap; increase declared cap before selection")
    def resolve(point):
        return unique_cases([{**point["case"],"inner_repeats":repeats}],p["search"]["k"])[0]
    values=unique_cases([resolve(x) for x in selected],p["search"]["k"]);targets=[]
    for method,target,neighbors in provisional:
        ids=[resolve(x)["case_config_id"] for x in neighbors]
        targets.append({"method":method,"target":target,"anchor":ids[0],"upper_neighbor":ids[1] if len(ids)>1 else None})
    # Capacity-only check on future test size; no test query computation or performance access.
    m=p["training"]["m"];bins=1<<p["training"]["nbits"];d=data["dimension"];nc=p["training"]["coarse_centers"]
    strategy=values[0];window=nq if strategy["prepare_window"]=="all" else min(nq,strategy["prepare_window"])
    chunk=window if strategy["compute_chunk"]=="all" else min(window,strategy["compute_chunk"])
    capacity_estimate=4*window*(m*bins+nc)+4*chunk*(3*d+nc)+32*data["n_queries"]
    if capacity_estimate>p["selection"]["scratch_budget_bytes"]:raise ValueError("test-window scratch capacity exceeds preregistered budget")
    evidence=[identity(Path(x)/"complete.json" if Path(x).is_dir() else x) for x in a.runs]
    doc={"schema_version":2,"phase":"test","dataset_id":data["dataset_id"],"bindings":bindings,
         "frozen":True,"local_smoke":smoke,"cases":values,"targets":targets,
         "common_inner_repeats":repeats,"test_scratch_capacity_estimate_bytes":capacity_estimate,
         "method_selection":identity(a.method_selection) if a.method_selection else None,
         "recommended_method":recommendation["recommended_method"] if recommendation else "not_evaluated_smoke",
         "rule":"use anchor; if its measured recall misses target, use the frozen upper neighbor if qualified; otherwise NOT_REACHED",
         "evidence":evidence}
    destinations=[a.out,a.diagnostic_out,a.mechanism_out]
    if any(x and Path(x).exists() for x in destinations):raise ValueError("freeze outputs are immutable")
    mechanism=None
    if a.mechanism_out:
        dev,_=rows(a.dev_runs,"dev",bindings);development=aggregate(dev)
        hnsw=[x for x in development if x["case"]["method"]=="hnsw"]
        if not hnsw:raise ValueError("mechanism freeze requires HNSW development results")
        beta=p["selection"]["mechanism_beta"]
        if beta not in p["development"]["beta_grid"]:raise ValueError("mechanism beta must belong to the shared dev grid")
        ef_values=sorted({min(hnsw,key=lambda x:abs(x["recall_at_k"]-target))["case"]["ef_search"]
                          for target in (.97,.98)})
        configs=[{"method":method,"ef_search":ef,"beta":beta,"prepare_window":strategy["prepare_window"],
                  "compute_chunk":strategy["compute_chunk"],"inner_repeats":1}
                 for method in ("opq","ivf_opq") for ef in ef_values]
        mechanism={**doc,"phase":"mechanism","cases":unique_cases(configs,p["search"]["k"]),
                   "query_ids":read_ids(data["splits"]["select"]["ids"]["path"])[:1000],
                   "targets":[],"common_inner_repeats":1,
                   "dev_evidence":[identity(Path(x)/"complete.json" if Path(x).is_dir() else x) for x in a.dev_runs],
                   "mechanism_rule":"shared beta; HNSW dev ef nearest 97% and 98%; select queries only"}
    seal(a.out,doc)
    if a.diagnostic_out:
        wanted={t["anchor"] for t in targets if t["target"] in (.97,.98)}
        if not wanted:wanted={t["anchor"] for t in targets}
        configs=[{**c,"inner_repeats":1,"source_test_case_id":c["case_config_id"]} for c in values if c["case_config_id"] in wanted]
        seal(a.diagnostic_out,{**doc,"phase":"diagnostic","cases":unique_cases(configs,p["search"]["k"]),
                               "parent_freeze":identity(a.out),"common_inner_repeats":1})
    if mechanism:seal(a.mechanism_out,{**mechanism,"parent_freeze":identity(a.out)})
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--runs",nargs="+",required=True)
    p.add_argument("--out",required=True);p.add_argument("--diagnostic-out");p.add_argument("--mechanism-out")
    p.add_argument("--dev-runs",nargs="*",default=[]);p.add_argument("--method-selection");freeze(p.parse_args())
if __name__=="__main__":cli(main)
