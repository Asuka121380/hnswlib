"""Validate actual timed top-k output before publishing any completed case."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
from .common import cli,load,unseal,verify_file
from .query_contract import evaluate_results

def validate(directory,case,truth,k,labels,metrics_level):
    directory=Path(directory);m=load(directory/"performance.json")
    n=len(truth);r=case["inner_repeats"]
    expected={"schema_version":2,"method":case["method"],"ef_search":case["ef_search"],
      "prepare_window":case["prepare_window"],"compute_chunk":case["compute_chunk"],
      "inner_repeats":r,"n_unique_queries":n,"n_query_executions":n*r,
      "metrics_level":metrics_level,"explicit_warmup":True,"timed_results_written":True,
      "timing_instrumented":False,"results_stable_across_repeats":True}
    for key,val in expected.items():
        if m.get(key)!=val:raise ValueError(f"runner contract mismatch {key}: {m.get(key)} != {val}")
    if m.get("beta")!=case["beta"]:raise ValueError("beta mismatch")
    wall=m["service_wall_ns"]
    if not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<=0:raise ValueError("invalid service wall")
    if sum(m[x] for x in ("prepare_wall_ns","search_and_materialize_wall_ns","other_wall_ns"))!=wall:
        raise ValueError("timing component mismatch")
    if not math.isclose(m["qps_service"],n*r*1e9/wall,rel_tol=1e-9):raise ValueError("QPS denominator mismatch")
    if len(m["repeat_metrics"])!=r or sum(x["service_wall_ns"] for x in m["repeat_metrics"])!=wall:
        raise ValueError("inner repeat coverage mismatch")
    if case["method"]!="hnsw" and "blas" not in m["batch_engine"].lower():
        raise ValueError("quantizer is not using the required BLAS engine")
    if metrics_level=="off":
        if m.get("exact_l2_calls") is not None or m.get("attempted_estimates") is not None:raise ValueError("instrumented primary timing")
    else:
        if m.get("catalog_mismatch") or m.get("backend_exception"):raise ValueError("unexpected estimator fallback")
        if not isinstance(m.get("exact_l2_calls"),int) or m["exact_l2_calls"]<=0:raise ValueError("missing actual L2 counts")
    quality=evaluate_results(directory/"results.csv",truth,k,r,set(map(int,labels)))
    if not quality["results_stable_across_repeats"]:raise ValueError("non-deterministic repeated results")
    return {**m,**quality}

def read_completed(path):
    doc=unseal(path)
    for key in ("run_manifest","query_order"):verify_file(doc["identity"][key])
    for f in doc["outputs"]:verify_file(f)
    return doc

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("complete");a=p.parse_args()
    print(read_completed(a.complete)["case_config_id"])
if __name__=="__main__":cli(main)
