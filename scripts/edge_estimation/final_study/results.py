"""Read only hash-verified published runs and aggregate all repeated measurements."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import identity,verify_file,unseal,load
from .validate_run import read_completed

def rows(paths,phase=None,bindings=None):
    result=[];seen=set();manifests=[]
    for value in paths:
        root=Path(value);complete=root/"complete.json" if root.is_dir() else root
        doc=unseal(complete);manifest=unseal(verify_file(doc["run_manifest"]))
        for entry in manifest["bindings"].values():verify_file(entry)
        if phase and manifest["phase"]!=phase:raise ValueError("wrong evidence phase")
        if bindings and manifest["bindings"]!=bindings:raise ValueError("evidence from different inputs")
        manifests.append(manifest)
        for entry in doc["cases"]:
            casefile=verify_file(entry);case=read_completed(casefile)
            if case["identity"]["run_manifest"]!=doc["run_manifest"]:raise ValueError("case/run identity mismatch")
            metrics=next((f for f in case["outputs"] if Path(f["path"]).name=="metrics.json"),None)
            if not metrics:raise ValueError("missing validated metrics")
            row=load(verify_file(metrics))
            key=(row["phase"],row["allocation_id"],row["block_id"],row["case_config_id"])
            if key in seen:raise ValueError("duplicate allocation/block/case evidence")
            seen.add(key);result.append(row)
        expected=unseal(verify_file(manifest["cases"]))["cases"]
        wanted={(manifest["phase"],manifest["allocation_id"],b,c["case_config_id"])
                for b in range(manifest["blocks"]) for c in expected}
        actual={(r["phase"],r["allocation_id"],r["block_id"],r["case_config_id"]) for r in result if
                r["phase"]==manifest["phase"] and r["allocation_id"]==manifest["allocation_id"]}
        if actual!=wanted:raise ValueError("published run case/block coverage mismatch")
    if not result:raise ValueError("no completed measurements")
    return result,manifests

def aggregate(values):
    groups={}
    for row in values:groups.setdefault(row["case_config_id"],[]).append(row)
    out=[]
    for key,group in sorted(groups.items()):
        first=group[0];point={"case_config_id":key,"case":first["case"],"count":len(group)}
        for field in first:
            cells=[r.get(field) for r in group]
            if all(type(v) in (int,float) for v in cells):
                point[field]=float(np.median(cells))
                point[field+"_mean"]=float(np.mean(cells))
                point[field+"_std"]=float(np.std(cells,ddof=1)) if len(cells)>1 else 0.0
                point[field+"_min"]=float(min(cells));point[field+"_max"]=float(max(cells))
        q=np.array([r["qps_service"] for r in group])
        point["qps_cv"]=float(q.std(ddof=1)/q.mean()) if len(q)>1 else 0.0
        # Recall over independent queries, not repeated executions.
        if any(r["per_query"]!=first["per_query"] for r in group):raise ValueError("query results vary across blocks")
        point["per_query"]=first["per_query"];out.append(point)
    return out
