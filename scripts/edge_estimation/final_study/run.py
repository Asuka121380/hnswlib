"""Serial, paired, identity-checked final-study measurement driver."""
from __future__ import annotations
import argparse, os, random, subprocess, time
from pathlib import Path
import numpy as np
from .common import cli,load,write,seal,unseal,identity,verify_file,digest,seed,name
from .schema import protocol,unique_cases
from .environment import verify_build,runtime
from .affinity import check_current_cpu
from .asset_contract import verify_assets
from .query_contract import split_contract,read_ids,write_ids
from .validate_run import validate,read_completed

def execute(a):
    p=protocol(a.protocol);doc=unseal(a.cases);phase=doc["phase"]
    if phase not in ("dev","batch","select","test","diagnostic","mechanism"):raise ValueError("invalid phase")
    data=unseal(a.dataset);assets=verify_assets(a.assets);build=verify_build(a.build,not a.local_smoke)
    if not data.get("validated"):raise ValueError("unvalidated dataset")
    if data.get("synthetic_only") and not a.local_smoke:
        raise ValueError("synthetic dataset is restricted to --local-smoke")
    if data.get("preprocessing_manifest"):verify_file(data["preprocessing_manifest"])
    for key,val in p["training"].items():
        actual=assets["training_spec"].get(key)
        if key=="trainer":
            if any(actual.get(k)!=v for k,v in val.items()):raise ValueError("asset trainer differs from study protocol")
        elif actual!=val:raise ValueError("asset training shape/budget differs from study protocol: "+key)
    if data["dataset_id"]!=doc["dataset_id"] or assets["dataset"]["sha256"]!=identity(a.dataset)["sha256"]:
        raise ValueError("dataset binding mismatch")
    bindings={k:identity(getattr(a,k)) for k in ("protocol","dataset","assets","build")}
    if doc["bindings"]!=bindings:raise ValueError("cases/freeze input identities differ")
    for entry in doc.get("evidence",[]):verify_file(entry)
    for key in ("parent_freeze","method_selection"):
        if doc.get(key):verify_file(doc[key])
    if phase in ("test","diagnostic","mechanism") and not doc.get("frozen"):raise ValueError("test/diagnostic requires an immutable freeze")
    if doc.get("local_smoke") and not a.local_smoke:
        raise ValueError("smoke-selected parameters cannot become a formal freeze")
    cases=unique_cases(doc["cases"],p["search"]["k"])
    if [c["case_config_id"] for c in cases]!=[c["case_config_id"] for c in doc["cases"]]:
        raise ValueError("case IDs differ from their configurations")
    split="dev" if phase=="dev" else "test" if phase in ("test","diagnostic") else "select"
    ids,truth=split_contract(data,split,p["search"]["k"])
    if phase=="mechanism":
        selected=doc["query_ids"]
        if not selected or len(selected)>1000 or len(set(selected))!=len(selected) or not set(selected).issubset(ids):
            raise ValueError("mechanism queries must be a frozen select subset of at most 1000")
        ids=selected;truth={q:truth[q] for q in ids}
    warm=read_ids(verify_file(data["warmup_ids"]),data["n_queries"])
    # The reserved warmup IDs belong to dev, but are never measured in a dev sweep.
    if phase=="dev":ids=[q for q in ids if q not in set(warm)];truth={q:truth[q] for q in ids}
    if not ids or set(ids)&set(warm):raise ValueError("empty measurement pool or warmup overlap")
    labels=np.load(verify_file(data["base_labels"]),allow_pickle=False)
    runtime_doc=runtime(not a.local_smoke)
    if not a.local_smoke and (p["measurement"]["allocations"]!=3 or p["measurement"]["blocks_per_allocation"]!=3):
        raise ValueError("formal protocol requires 3 allocations x 3 blocks")
    if a.allocation<0 or a.allocation>=p["measurement"]["allocations"]:raise ValueError("allocation outside protocol")
    blocks=a.blocks if a.blocks is not None else 1 if phase in ("diagnostic","mechanism") else p["measurement"]["blocks_per_allocation"]
    if blocks<=0 or (phase=="test" and blocks!=p["measurement"]["blocks_per_allocation"]):
        raise ValueError("invalid block count")
    if phase in ("diagnostic","mechanism") and (blocks!=1 or a.allocation!=0 or any(c["inner_repeats"]!=1 for c in cases)):
        raise ValueError("diagnostics require one pass, one block, allocation 0")
    if phase in ("batch","select") and blocks not in (3,6,9) and not a.local_smoke:raise ValueError("selection requires 3, 6 or 9 blocks")
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    expected={"schema_version":2,"phase":phase,"dataset_id":data["dataset_id"],"study_id":p["study_id"],
      "bindings":bindings,"cases":identity(a.cases),"allocation_id":a.allocation,
      "blocks":blocks,"local_smoke":a.local_smoke,"slurm_job_id":runtime_doc["slurm_job_id"],
      "hostname":runtime_doc["hostname"],"runtime":runtime_doc}
    manifest=out/"run_manifest.json"
    if manifest.exists():
        old=unseal(manifest)
        # Process lists/frequency can change; stable environment identifiers cannot.
        for key in expected:
            if key!="runtime" and old[key]!=expected[key]:raise ValueError("resume identity mismatch: "+key)
        for key in ("affinity","thread_environment","machine"):
            if old["runtime"][key]!=runtime_doc[key]:raise ValueError("resume environment mismatch")
    else:seal(manifest,expected)
    lock=out/"measurement.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    completed=[]
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        for block in range(blocks):
            blockdir=out/f"block-{block:02d}";blockdir.mkdir(exist_ok=True)
            query_phase="test" if phase=="diagnostic" else phase
            order=list(ids);random.Random(seed(p["study_id"],data["dataset_id"],query_phase,a.allocation,block,"queries")).shuffle(order)
            qpath=blockdir/"query_ids.txt"
            if qpath.exists():
                if read_ids(qpath)!=order:raise ValueError("resume query order changed")
            else:write_ids(qpath,order)
            sequence=list(cases);random.Random(seed(p["study_id"],data["dataset_id"],phase,a.allocation,block,"cases")).shuffle(sequence)
            for ordinal,c in enumerate(sequence):
                dest=blockdir/c["case_config_id"];casefile=dest/"complete.json"
                run_id={"case_config_id":c["case_config_id"],"case":c,"allocation_id":a.allocation,
                    "block_id":block,"ordinal":ordinal,"phase":phase,"dataset_id":data["dataset_id"],
                    "run_manifest":identity(manifest),"query_order":identity(qpath)}
                if casefile.exists():
                    previous=read_completed(casefile)
                    if previous["identity"]!=run_id:raise ValueError("completed case identity mismatch")
                    completed.append(identity(casefile));continue
                if dest.exists():raise ValueError(f"incomplete case at {dest}; preserve it and rerun in a new allocation directory")
                dest.mkdir()
                runner=build["files"]["uq_ivf_performance_runner" if c["method"]=="ivf_opq" else "uq_opq_performance_runner"]["path"]
                mode="diagnostic" if phase in ("diagnostic","mechanism") else "off"
                cmd=[runner,"--index-path",assets["index"]["path"],"--query-path",data["query_pool"]["path"],
                    "--query-ids",str(qpath),"--warmup-query-ids",data["warmup_ids"]["path"],
                    "--dimension",str(data["dimension"]),"--k",str(p["search"]["k"]),"--ef-search",str(c["ef_search"]),
                    "--prepare-window",str(c["prepare_window"]),"--compute-chunk",str(c["compute_chunk"]),
                    "--repeats",str(c["inner_repeats"]),"--metrics-level",mode,
                    "--max-result-bytes",str(p["measurement"]["result_buffer_cap_bytes"]),
                    "--output",str(dest/"performance.json"),"--result-records-output",str(dest/"results.csv")]
                if c["method"]=="hnsw":cmd+=["--no-prune"]
                else:cmd+=["--artifact-path",assets["artifacts"][c["method"]]["path"],"--beta",str(c["beta"])]
                write(dest/"command.json",cmd)
                if not a.local_smoke:check_current_cpu(runtime_doc["affinity"])
                with (dest/"run.log").open("w",encoding="utf-8") as log:
                    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:raise ValueError(f"runner failed; see {dest/'run.log'}")
                metrics=validate(dest,c,truth,p["search"]["k"],labels,mode)
                write(dest/"metrics.json",{**metrics,**run_id,"local_smoke":a.local_smoke})
                outputs=[identity(f) for f in sorted(dest.iterdir()) if f.is_file()]
                seal(casefile,{"schema_version":2,"identity":run_id,"case_config_id":c["case_config_id"],"outputs":outputs})
                completed.append(identity(casefile))
                print(f"{phase} allocation={a.allocation} block={block} {c['method']} ef={c['ef_search']} recall={metrics['recall_at_k']:.5f} QPS={metrics['qps_service']:.2f}",flush=True)
        # Detect source/input mutation during the allocation before publication.
        verify_assets(a.assets);verify_build(a.build,not a.local_smoke)
        for entry in bindings.values():verify_file(entry)
        if not a.local_smoke:check_current_cpu(runtime_doc["affinity"])
        seal(out/"complete.json",{"schema_version":2,"run_manifest":identity(manifest),"cases":completed})
    finally:
        if lock.exists():lock.unlink()
    return out/"complete.json"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("protocol","dataset","assets","build","cases","out"):p.add_argument("--"+key,required=True)
    p.add_argument("--allocation",type=int,default=0);p.add_argument("--blocks",type=int)
    p.add_argument("--local-smoke",action="store_true");execute(p.parse_args())
if __name__=="__main__":cli(main)
