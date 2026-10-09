"""Account for offline work, resident-memory measurements and edge payload costs."""
from __future__ import annotations
import argparse
from pathlib import Path
from .common import cli,load,unseal,seal,identity,verify_file
from .asset_contract import verify_assets
from .analyze import table

def report(a):
    assets=verify_assets(a.assets);analysis=unseal(a.analysis)
    if unseal(verify_file(analysis["freeze"]))["bindings"]["assets"]!=identity(a.assets):
        raise ValueError("analysis/asset mismatch")
    opq=Path(assets["artifacts"]["opq"]["path"]);cr=Path(assets["artifacts"]["ivf_opq"]["path"])
    t=load(opq/"training.json");m=load(cr/"manifest.json");coarse=m["coarse_manifest"]
    offline={"opq":{"quantizer_training_seconds":t["training_seconds"],"encoding_seconds":t["encoding_seconds"]},
             "ivf_opq":{"coarse_training_seconds":coarse["coarse_training_seconds"],
                "coarse_assignment_seconds":coarse["assignment_seconds"],
                "quantizer_training_seconds":m["quantizer_training_seconds"],"encoding_seconds":m["encoding_seconds"]}}
    sizes={}
    for method,art in assets["artifacts"].items():
        files=art["files"]
        record=next(f for f in files if Path(f["path"]).name=="edges.bin")
        sizes[method]={"all_artifact_file_bytes":sum(f["bytes"] for f in files),
            "edge_record_bytes":record["bytes"],"bytes_per_edge":record["bytes"]/assets["edge_count"]}
    amortization=[]
    for t in analysis["targets"]:
        if t["method"]=="hnsw":continue
        baseline=next(r for r in analysis["targets"] if r["method"]=="hnsw" and r["target"]==t["target"])
        saved=1/baseline["qps_service"]-1/t["qps_service"]
        amortization.append({"method":t["method"],"target":t["target"],
          "extra_quantizer_offline_seconds":sum(offline[t["method"]].values()),
          "queries_to_amortize":sum(offline[t["method"]].values())/saved
            if saved>0 and t["status"]==baseline["status"]=="OK" else None,
          "note":"excludes shared graph build, sampling, I/O validation; full wall ledger reported separately"})
    ledger=unseal(a.ledger) if a.ledger else None
    if ledger and ledger["asset_contract"]!=identity(a.assets):raise ValueError("offline ledger asset mismatch")
    points={p["case_config_id"]:p for p in analysis["points"]};online=[]
    for target in analysis["targets"]:
        point=points[target["selected_case"]]
        online.append({"method":target["method"],"target":target["target"],"case_config_id":target["selected_case"],
            **{k:point.get(k) for k in ("query_bytes","result_buffer_bytes","backend_bytes","catalog_payload_bytes",
               "scratch_bytes","peak_rss_bytes","peak_rss_after_graph_bytes","peak_rss_after_backend_bytes",
               "peak_rss_after_warmup_bytes","graph_load_ns","backend_file_hash_ns",
               "backend_load_and_structural_validation_ns","prepare_wall_ns","search_and_materialize_wall_ns",
               "service_wall_ns","n_query_executions")}})
    seal(a.out,{"schema_version":2,"analysis":identity(a.analysis),"assets":identity(a.assets),
        "offline_components":offline,"offline_wall_ledger":ledger,"storage":sizes,"amortization":amortization,
        "online_costs":online,
        "memory_note":"Use measured peak_rss_bytes for process peaks; artifact file bytes are storage, not RSS. Catalog/query/result/scratch bytes remain separate in points."})
    table(str(Path(a.out).with_suffix(""))+".online.csv",online)
    table(str(Path(a.out).with_suffix(""))+".offline.csv",[{"method":m,**v,**sizes[m]} for m,v in offline.items()])
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("assets","analysis","out"):p.add_argument("--"+key,required=True)
    p.add_argument("--ledger");report(p.parse_args())
if __name__=="__main__":cli(main)
