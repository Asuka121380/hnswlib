"""Frozen-target reporting with paired allocation/block and independent-query uncertainty."""
from __future__ import annotations
import argparse,math,csv,json
from pathlib import Path
import numpy as np
from .common import cli,unseal,identity,seal,verify_file,seed
from .schema import protocol
from .results import rows,aggregate

def paired_interval(numerator,denominator,rng,draws=2000):
    x={(r["allocation_id"],r["block_id"]):r["qps_service"] for r in numerator}
    y={(r["allocation_id"],r["block_id"]):r["qps_service"] for r in denominator}
    if set(x)!=set(y):raise ValueError("unpaired allocation/block coverage")
    allocations=sorted({a for a,b in x});samples=[]
    for _ in range(draws):
        ratios=[]
        for a in rng.choice(allocations,len(allocations),replace=True):
            blocks=sorted(b for aa,b in x if aa==a)
            for b in rng.choice(blocks,len(blocks),replace=True):ratios.append(math.log(x[a,b]/y[a,b]))
        samples.append(math.exp(float(np.mean(ratios))))
    point=math.exp(float(np.mean([math.log(x[k]/y[k]) for k in x])))
    return {"paired_geomean_speedup":point,"speedup_ci95":list(map(float,np.quantile(samples,[.025,.975]))),
            "bootstrap_units":"allocations_then_paired_blocks","allocation_count":len(allocations)}

def query_interval(values,rng,draws=2000):
    x=np.array([v["recall"] for v in values])
    samples=[float(rng.choice(x,len(x),replace=True).mean()) for _ in range(draws)]
    return list(map(float,np.quantile(samples,[.025,.975])))

def paired_query_samples(points,rng,draws):
    keys=list(points);ids=[q["query_id"] for q in points[keys[0]]["per_query"]]
    if any([q["query_id"] for q in points[key]["per_query"]]!=ids for key in keys):
        raise ValueError("recall bootstrap requires paired unique query IDs")
    matrix=np.array([[q["recall"] for q in points[key]["per_query"]] for key in keys])
    samples=[]
    for first in range(0,draws,64):
        indices=rng.integers(0,len(ids),size=(min(64,draws-first),len(ids)))
        samples.append(matrix[:,indices].mean(axis=2))
    values=np.concatenate(samples,axis=1)
    return {key:values[i] for i,key in enumerate(keys)}

def table(path,values):
    if not values:return
    fields=sorted({k for row in values for k in row if k!="per_query"})
    with Path(path).open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in values:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
                             for k,v in row.items() if k in fields})

def analyze(a):
    frozen=unseal(a.freeze)
    if not frozen.get("frozen") or frozen["phase"]!="test":raise ValueError("test freeze required")
    p=protocol(frozen["bindings"]["protocol"]["path"])
    raw,manifests=rows(a.runs,"test",frozen["bindings"])
    expected=identity(a.freeze)
    if any(m["cases"]!=expected for m in manifests):raise ValueError("run does not use this freeze")
    smoke=any(m["local_smoke"] for m in manifests)
    allocations={m["allocation_id"] for m in manifests}
    if not smoke:
        if allocations!=set(range(p["measurement"]["allocations"])):raise ValueError("all three independent allocations required")
        jobs={m["slurm_job_id"] for m in manifests}
        if None in jobs or len(jobs)!=len(allocations):raise ValueError("allocations must come from distinct Slurm jobs")
        cpu=manifests[0]["runtime"]["cpu_contract"]
        if any(m["runtime"]["cpu_contract"]!=cpu for m in manifests):raise ValueError("mixed CPU model/ISA in formal comparison")
    points={r["case_config_id"]:r for r in aggregate(raw)}
    rng=np.random.default_rng(seed(p["study_id"],frozen["dataset_id"],"analysis"));targets=[]
    recall_samples=paired_query_samples(points,rng,a.bootstrap_draws)
    for t in frozen["targets"]:
        chosen=points[t["anchor"]];used_neighbor=False
        if chosen["recall_at_k"]<t["target"] and t["upper_neighbor"]:
            neighbor=points[t["upper_neighbor"]]
            if neighbor["recall_at_k"]>=t["target"]:chosen=neighbor;used_neighbor=True
        reached=chosen["recall_at_k"]>=t["target"]
        interval=list(map(float,np.quantile(recall_samples[chosen["case_config_id"]],[.025,.975])))
        targets.append({**t,"selected_case":chosen["case_config_id"],"status":"OK" if reached else "NOT_REACHED",
            "used_upper_neighbor":used_neighbor,"recall_at_k":chosen["recall_at_k"],"recall_ci95":interval,
            "recall_lower_ci_meets_target":interval[0]>=t["target"],"qps_service":chosen["qps_service"],
            "speedup_ci95":None})
    for point in targets:
        baseline=next(t for t in targets if t["method"]=="hnsw" and t["target"]==point["target"])
        point["paired_recall_difference_ci95"]=list(map(float,np.quantile(
            recall_samples[point["selected_case"]]-recall_samples[baseline["selected_case"]],[.025,.975])))
        if point["status"]=="OK" and baseline["status"]=="OK":
            point.update(paired_interval([r for r in raw if r["case_config_id"]==point["selected_case"]],
                 [r for r in raw if r["case_config_id"]==baseline["selected_case"]],rng,a.bootstrap_draws))
    diagnostics=None
    if a.diagnostics:
        diagnostic_rows,diagnostic_manifests=rows(a.diagnostics,"diagnostic",frozen["bindings"])
        for manifest in diagnostic_manifests:
            if unseal(verify_file(manifest["cases"])).get("parent_freeze")!=expected:
                raise ValueError("diagnostic belongs to another freeze")
        diagnostics=aggregate(diagnostic_rows)
        for row in diagnostics:
            source=row["case"].get("source_test_case_id")
            if source not in points or row["per_query"]!=points[source]["per_query"]:
                raise ValueError("diagnostic and uninstrumented top-k differ")
            row["exact_l2_calls_per_query"]=row["exact_l2_calls"]/row["n_query_executions"]
            attempted=row.get("attempted_estimates")
            row["prune_rate"]=row["pruned_estimates"]/attempted if attempted else None
    mechanisms=None
    if a.mechanisms:
        mechanism_rows,mechanism_manifests=rows(a.mechanisms,"mechanism",frozen["bindings"])
        for manifest in mechanism_manifests:
            contract=unseal(verify_file(manifest["cases"]))
            if contract.get("parent_freeze")!=expected:raise ValueError("mechanism belongs to another freeze")
        mechanisms=aggregate(mechanism_rows)
        for row in mechanisms:
            row["exact_l2_calls_per_query"]=row["exact_l2_calls"]/row["n_query_executions"]
            row["diagnostic_qps_not_for_speed_claims"]=True
    if Path(a.out).exists():raise ValueError("analysis output exists")
    seal(a.out,{"schema_version":2,"dataset_id":frozen["dataset_id"],"local_smoke":smoke,
        "freeze":expected,"runs":[identity(Path(x)/"complete.json" if Path(x).is_dir() else x) for x in a.runs],
        "targets":targets,"points":list(points.values()),"diagnostics":diagnostics,"mechanisms":mechanisms,
        "uncertainty_note":"Recall CI resamples unique queries. QPS CI resamples allocations, then paired blocks; inner repeats are not independent samples.",
        "minimum_observed_case_seconds":min(r["service_wall_ns"]/1e9 for r in raw)})
    stem=Path(a.out).with_suffix("")
    exports={"all_cases":raw,"registered_curve_summary":list(points.values()),
             "frozen_anchor_table":targets,"diagnostic":diagnostics or [],"mechanism":mechanisms or []}
    for name,values in exports.items():table(str(stem)+"."+name+".csv",values)
    seal(str(stem)+".reproduction_manifest.json",{"analysis":identity(a.out),"freeze":expected,
        "tables":[identity(str(stem)+"."+name+".csv") for name,values in exports.items() if values],
        "runs":[identity(Path(x)/"complete.json" if Path(x).is_dir() else x) for x in a.runs],
        "diagnostic_runs":[identity(Path(x)/"complete.json" if Path(x).is_dir() else x) for x in (a.diagnostics or [])],
        "mechanism_runs":[identity(Path(x)/"complete.json" if Path(x).is_dir() else x) for x in (a.mechanisms or [])]})
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--freeze",required=True)
    p.add_argument("--runs",nargs="+",required=True);p.add_argument("--diagnostics",nargs="*")
    p.add_argument("--mechanisms",nargs="*")
    p.add_argument("--out",required=True);p.add_argument("--bootstrap-draws",type=int,default=10000)
    a=p.parse_args()
    if a.bootstrap_draws<100:raise ValueError("need at least 100 bootstrap draws")
    analyze(a)
if __name__=="__main__":cli(main)
