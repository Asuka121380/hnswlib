"""Train, calibrate, benchmark and summarize coarse-residual HNSW methods."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.edge_estimation.ivf_train import dump, verify
from scripts.edge_estimation.contracts import sha256_file, read_events
from scripts.edge_estimation.run_recall_qps_grid import read_ground_truth, recall_at_k, command_for as legacy_command

ROOT = Path(__file__).resolve().parents[2]


def execute(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    dump(log.with_suffix(".command.json"), list(map(str,command)))
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(list(map(str,command)), check=True, stdout=stream, stderr=subprocess.STDOUT)


def table(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def config(path):
    data=json.loads(path.read_text())
    if data.get("architecture") != "hnsw_coarse_residual_edges":
        raise ValueError("this runner has efSearch, not nprobe/list routing")
    for key in ("dimension","m","nbits","query_count","repeats","batch_size","sample_cap"):
        if int(data[key])<=0: raise ValueError(f"invalid {key}")
    if data["dimension"] % data["m"] or data["nbits"]>8:
        raise ValueError("invalid quantizer shape")
    for method,grid in data["methods"].items():
        if method not in {"hnsw","ivf_pq","ivf_opq","ivf_pq_qjl","pq8","pq_qjl","opq"}: raise ValueError("unsupported method")
        if not grid["ef_search"] or min(grid["ef_search"])<data["k"]: raise ValueError("invalid efSearch")
        if method != "hnsw" and (not grid["beta"] or any(not np.isfinite(b) or b<=0 for b in grid["beta"])):
            raise ValueError("invalid beta")
    return data


def identity(path):
    return {"path":str(path.resolve()),"sha256":sha256_file(path),"size":path.stat().st_size}


def expanded_cases(cfg):
    """Expand either a Cartesian grid or an explicit, unique list of configurations."""
    templates = cfg.get("cases")
    if templates is None:
        templates=[]
        for method, params in cfg["methods"].items():
            batches = [1] if method in {"hnsw","pq8","pq_qjl"} else cfg.get("batch_grid",[cfg["batch_size"]])
            for ef,beta,batch in itertools.product(params["ef_search"],params.get("beta",[None]),batches):
                templates.append(dict(method=method,ef_search=ef,beta=beta,batch_size=batch))
    if not templates:
        raise ValueError("empty case list")
    seen=set()
    for case in templates:
        method=case["method"]
        if method not in cfg["methods"] or case["ef_search"] < cfg["k"] or case["batch_size"] < 1:
            raise ValueError("invalid explicit case")
        if method != "hnsw" and (case["beta"] is None or not np.isfinite(case["beta"]) or case["beta"] <= 0):
            raise ValueError("invalid explicit beta")
        if method in {"hnsw","pq8","pq_qjl"} and case["batch_size"] != 1:
            raise ValueError("method only supports batch=1")
        key=tuple(case[k] for k in ("method","ef_search","beta","batch_size"))
        if key in seen: raise ValueError("duplicate explicit case")
        seen.add(key)
    cases=[]
    for repeat in range(cfg["repeats"]):
        block=[{**case,"repeat_id":repeat} for case in templates]
        random.Random(cfg.get("seed",42)+repeat).shuffle(block)
        cases.extend(block)
    return cases


def train(args,cfg):
    args.out.mkdir(parents=True, exist_ok=True)
    script=ROOT / "scripts/edge_estimation/ivf_train.py"
    common=["--assets",args.assets,"--catalog",args.catalog,"--iterations",cfg.get("iterations",25),
            "--threads",cfg.get("training_threads",1),"--batch-size",cfg.get("encode_batch_size",4096)]
    for nlist in cfg["coarse_centers"]:
        dest=args.out/f"k{nlist}"; dest.mkdir()
        command=[sys.executable,script,"coarse",*common,"--dimension",cfg["dimension"],"--nlist",nlist,
                 "--seed",cfg.get("seed",42),"--sample-cap",cfg["sample_cap"],"--out",dest/"coarse"]
        if args.training_assets:
            command += ["--training-assets",args.training_assets,"--training-catalog",args.training_catalog]
        execute(command,dest/"coarse.log")
        for method in ("pq","opq","pq_qjl"):
            command=[sys.executable,script,"encode",*common,"--coarse",dest/"coarse","--method",method,
                     "--m",cfg["m"],"--nbits",cfg["nbits"],"--qjl-bits",cfg.get("qjl_bits",128),
                     "--qjl-seed",cfg.get("qjl_seed",43),"--opq-iterations",cfg.get("opq_iterations",25),
                     "--out",dest/f"ivf_{method}"]
            if method=="pq_qjl": command += ["--pq-artifact",dest/"ivf_pq"]
            execute(command,dest/f"{method}.log")
    dump(args.out/"training_complete.json", {"config":cfg,"assets":identity(args.assets)})


def quality(args,cfg):
    args.out.mkdir(parents=True,exist_ok=False)
    _,events=read_events(args.trace/"events.bin")
    ids=sorted({event.query_id for event in events})
    evaluation_ids=set(range(cfg.get("query_start",0),cfg.get("query_start",0)+cfg["query_count"]))
    overlap=bool(set(ids)&evaluation_ids)
    if overlap and not cfg.get("exploratory",False):
        raise ValueError("calibration/evaluation query overlap; use a separate split or mark exploratory")
    rows=[]
    for method, grid in cfg["methods"].items():
        if not method.startswith("ivf_"): continue
        artifact=args.artifacts/method; verify(artifact)
        for beta in grid["beta"]:
            value=subprocess.run([str(args.native),"quality",str(args.trace/"events.bin"),
                str(args.trace/"labels.bin"),str(artifact),str(args.queries),str(beta)],
                check=True,capture_output=True,text=True)
            report=json.loads(value.stdout)
            dump(args.out/f"{method}-{beta}.json",report)
            c=report
            den=report["decision_count_s"]
            query_rates=[q["counts"]["fp"]/q["counts"]["decision_count_s"]
                         for q in report["per_query"] if q["counts"]["decision_count_s"]>0]
            if not query_rates or not den: raise ValueError("quality trace has no decisions")
            fp=c["fp"]/max(1,den); p95=float(np.quantile(query_rates,.95))
            rows.append({"method":method,"beta":beta,"correct_prune_rate":c["tp"]/max(1,den),
                         "false_prune_rate":fp,"conditional_false_prune_rate":c["fp"]/max(1,c["fp"]+c["tn"]),
                         "p95_query_false_prune_rate":p95,"decisions":den,
                         "balanced":fp<=.005 and p95<=.01,"strict":fp<=.001 and p95<=.0025})
    table(args.out/"quality_sweep.csv",rows)
    selected=[]
    for method in {r["method"] for r in rows}:
        for profile in ("balanced","strict"):
            candidates=[r for r in rows if r["method"]==method and r[profile]]
            if candidates: selected.append({**max(candidates,key=lambda r:r["correct_prune_rate"]),"profile":profile})
    table(args.out/"selected.csv",selected)
    dump(args.out/"manifest.json",{"config":cfg,"query_ids":ids,"evaluation_overlap":overlap,
         "queries":identity(args.queries),"events":identity(args.trace/"events.bin"),
         "labels":identity(args.trace/"labels.bin"),"native":identity(args.native)})


def grid(args,cfg):
    args.out.mkdir(parents=True,exist_ok=False)
    artifacts={method:verify(args.artifacts/method) for method in cfg["methods"] if method.startswith("ivf_")}
    identities={a["coarse_identity"] for a in artifacts.values()}
    if len(identities)!=1: raise ValueError("methods must share exactly one coarse bundle")
    first=next(iter(artifacts.values()))
    if first["assets"]["index"] != sha256_file(args.index): raise ValueError("index identity mismatch")
    for method,artifact in artifacts.items():
        if artifact["method"] != method or artifact["dimension"] != cfg["dimension"]:
            raise ValueError("method/dimension mismatch")
    truth=read_ground_truth(args.ground_truth,cfg.get("query_start",0),cfg["query_count"],cfg["k"])
    manifest={"config":cfg,"platform":platform.platform(),"python":sys.version,
        "inputs":{k:identity(getattr(args,k)) for k in ("index","queries","ground_truth","runner")},
        "artifacts":{m:identity(args.artifacts/m/"manifest.json") for m in artifacts},
        "thread_environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
        "started_unix":time.time()}
    dump(args.out/"manifest.json",manifest)
    control_inputs=set()
    if "opq" in cfg["methods"]: control_inputs.update(("opq_runner","opq_artifact"))
    if any(m in cfg["methods"] for m in ("pq8","pq_qjl")): control_inputs.update(("v0_runner","pq_sidecar"))
    if "pq_qjl" in cfg["methods"]: control_inputs.add("qjl_companion")
    for name in sorted(control_inputs):
        path=getattr(args,name)
        if path is None: raise ValueError(f"legacy control needs --{name.replace('_','-')}")
        manifest["inputs"][name]=identity(path/"manifest.json" if name=="opq_artifact" else path)
    dump(args.out/"manifest.json",manifest)
    cases=expanded_cases(cfg)
    dump(args.out/"cases.json",cases)
    rows=[]
    for ordinal,case in enumerate(cases):
        case_id=f"{ordinal:04d}-{case['method']}-ef{case['ef_search']}-b{case['beta']}-batch{case['batch_size']}-r{case['repeat_id']}"
        dest=args.out/case_id; dest.mkdir()
        method=case["method"]
        primary="ivf_pq" if "ivf_pq" in artifacts else next(iter(artifacts))
        artifact=args.artifacts/(primary if method=="hnsw" else method)
        cmd=[args.runner,"--index-path",args.index,"--artifact-path",artifact,"--query-path",args.queries,
             "--dimension",cfg["dimension"],"--query-start",cfg.get("query_start",0),"--query-count",cfg["query_count"],
             "--k",cfg["k"],"--ef-search",case["ef_search"],"--batch-size",case["batch_size"],
             "--warmup-queries",cfg.get("warmup_queries",100),"--repeats",1,"--output",dest/"performance.json",
             "--result-records-output",dest/"results.csv","--latency-records-output",dest/"latency.csv"]
        cmd += ["--no-prune"] if method=="hnsw" else ["--beta",case["beta"]]
        if method in {"pq8","pq_qjl","opq"}:
            cmd=legacy_command(case,args,{**cfg,"repeats":1,"opq_batch_size":case["batch_size"]},dest)
        execute(cmd,dest/"run.log")
        perf=json.loads((dest/"performance.json").read_text())
        row={**case,"case_id":case_id,"recall_at_k":recall_at_k(dest/"results.csv",truth,cfg["k"])}
        if not np.isfinite(perf["qps"]) or perf["qps"]<=0: raise ValueError("invalid runner timing")
        row.update({k:perf.get(k) for k in ("qps","latency_p50_ns","latency_p95_ns","latency_p99_ns",
            "batch_prepare_ns_per_query","batch_completion_service_p95_ns","attempted_estimates",
            "pruned_estimates","fallback_estimates","eligible_exact_distance_count","backend_bytes","scratch_bytes",
            "peak_rss_bytes","catalog_payload_bytes")})
        rows.append(row); table(args.out/"points.csv",rows)
        print(f"{ordinal+1}/{len(cases)} {case_id} recall={row['recall_at_k']:.5f} qps={row['qps']:.2f}",flush=True)
    summarize(args.out,args.plot)
    dump(args.out/"complete.json",{"points":len(rows),"elapsed_seconds":time.time()-manifest["started_unix"]})


def summarize(root,plot=False):
    with (root/"points.csv").open() as stream: rows=list(csv.DictReader(stream))
    for row in rows:
        for key in ("recall_at_k","qps"): row[key]=float(row[key])
    # Aggregate repeats of the same configuration before selecting a frontier.
    grouped={}
    for row in rows:
        key=tuple(row[k] for k in ("method","ef_search","beta","batch_size"))
        grouped.setdefault(key,[]).append(row)
    points=[]
    for group in grouped.values():
        point=dict(group[0]); qps=[r["qps"] for r in group]
        point.update(qps=float(np.median(qps)),qps_mean=float(np.mean(qps)),qps_std=float(np.std(qps)),
                     qps_min=min(qps),qps_max=max(qps),repeat_count=len(group),
                     recall_at_k=float(np.mean([r["recall_at_k"] for r in group])))
        points.append(point)
    table(root/"aggregated.csv",points)
    fronts=[]; thresholds=[]
    for method in sorted({r["method"] for r in points}):
        values=[r for r in points if r["method"]==method]
        fronts += [r for r in values if not any(s["recall_at_k"]>=r["recall_at_k"] and s["qps"]>=r["qps"]
                   and (s["recall_at_k"]>r["recall_at_k"] or s["qps"]>r["qps"]) for s in values)]
        for threshold in (.90,.95,.97,.98,.985):
            candidates=[r for r in values if r["recall_at_k"]>=threshold]
            thresholds.append({**(max(candidates,key=lambda r:r["qps"]) if candidates else {"method":method}),
                               "threshold":threshold,"status":"OK" if candidates else "NOT_REACHED"})
    table(root/"pareto.csv",fronts); table(root/"thresholds.csv",thresholds)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5))
        for method in sorted({r["method"] for r in fronts}):
            values=sorted([r for r in fronts if r["method"]==method],key=lambda r:r["recall_at_k"])
            ax.plot([r["recall_at_k"] for r in values],[r["qps"] for r in values],"o-",label=method)
        ax.set(xlabel="Recall@10",ylabel="QPS (preparation included)"); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(root/"recall_qps.png",dpi=180); fig.savefig(root/"recall_qps.pdf"); plt.close(fig)


def refine(args,cfg):
    """Derive bounded follow-ups from actual pilot points instead of assuming old betas transfer."""
    with (args.points/"pareto.csv").open() as stream: rows=list(csv.DictReader(stream))
    args.out.mkdir(parents=True,exist_ok=False)
    chosen={}
    for method in cfg["methods"]:
        candidates=[r for r in rows if r["method"]==method]
        if not candidates: raise ValueError(f"no completed pilot points for {method}")
        selected=[]
        for threshold in (.95,.98):
            qualified=[r for r in candidates if float(r["recall_at_k"])>=threshold]
            selected.append(max(qualified,key=lambda r:float(r["qps"])) if qualified else
                            max(candidates,key=lambda r:(float(r["recall_at_k"]),float(r["qps"]))))
        chosen[method]={"ef_search":sorted({int(r["ef_search"]) for r in selected})}
        if method!="hnsw": chosen[method]["beta"]=sorted({float(r["beta"]) for r in selected})
    for mode in ("high_recall","batch","confirm"):
        output=json.loads(json.dumps(cfg)); output["methods"]=json.loads(json.dumps(chosen))
        output["selection_source"]=identity(args.points/"points.csv")
        output.pop("batch_grid",None)
        output["repeats"]=3 if mode=="confirm" else 1
        if mode=="high_recall":
            for params in output["methods"].values(): params["ef_search"]=[950,1100,1300]
        if mode=="batch": output["batch_grid"]=[1,32,128,256]
        dump(args.out/f"{mode}.json",output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("train","quality","grid","summarize","refine"))
    p.add_argument("--config",type=Path)
    for key in ("assets","catalog","training-assets","training-catalog","artifacts","trace","queries","index","ground-truth","runner","native","points","out",
                "v0-runner","opq-runner","pq-sidecar","qjl-companion","opq-artifact"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--plot",action="store_true")
    args=p.parse_args()
    required={"train":["config","assets","catalog","out"],"quality":["config","artifacts","trace","queries","native","out"],
              "grid":["config","artifacts","index","queries","ground_truth","runner","out"],"summarize":["out"],
              "refine":["config","points","out"]}[args.stage]
    for name in required:
        if getattr(args,name) is None: p.error(f"{args.stage} requires --{name.replace('_','-')}")
    if bool(args.training_assets)!=bool(args.training_catalog): p.error("training inputs must be paired")
    if args.stage=="summarize": summarize(args.out,args.plot)
    else: globals()[args.stage](args,config(args.config))


if __name__ == "__main__": main()
