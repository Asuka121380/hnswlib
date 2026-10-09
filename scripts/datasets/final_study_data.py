"""Prepare auditable query pools and exhaustive GT. No implicit 'fresh query' inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.edge_estimation.final_study.common import identity, verify_file, load, seal, unseal, cli, name
from scripts.edge_estimation.final_study.query_contract import read_vecs, write_vecs, read_ids, write_ids
from scripts.edge_estimation.contracts import atomic_output_dir

def vectors(spec, dimension):
    path = Path(spec["path"])
    if spec.get("format","fvecs") == "fvecs":
        return read_vecs(path, "f", dimension)
    if spec["format"] != "fbin":
        raise ValueError("supported vector formats: fvecs/fbin")
    with path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError("truncated fbin")
    n, d = struct.unpack("<II", header)
    if d != dimension or not n or path.stat().st_size != 8+4*n*d:
        raise ValueError("fbin shape/length mismatch")
    values = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(n,d))
    for first in range(0, n, 65536):
        if not np.isfinite(values[first:first+65536]).all():
            raise ValueError("nonfinite fbin vector")
    return values

def vector_hash(row):
    value = np.array(row, dtype="<f4", copy=True)
    value[value == 0] = 0  # +0/-0 represent the same vector.
    return hashlib.sha256(value.tobytes()).hexdigest()

def registry(path):
    value = load(path)
    for dataset, spec in value["datasets"].items():
        name(dataset)
        if spec.get("metric","squared_l2") != "squared_l2":
            raise ValueError("only squared L2 is supported by this study")
        if spec["dimension"] <= 0 or spec["n_base"] <= 0:
            raise ValueError("invalid dataset shape")
    return value

def inventory(args):
    reg = registry(args.registry)
    datasets = {}
    for dataset, spec in reg["datasets"].items():
        sources = {}
        for key, source in {"base":spec["base"], **spec["query_sources"]}.items():
            values = vectors(source,spec["dimension"])
            norms = []
            for first in range(0,len(values),65536):
                block = values[first:first+65536]
                if not np.isfinite(block).all():
                    raise ValueError("nonfinite source vectors")
                norms.extend([float(np.min(np.linalg.norm(block,axis=1))),
                              float(np.max(np.linalg.norm(block,axis=1)))])
            sources[key] = {**identity(source["path"]), "format":source.get("format","fvecs"),
                            "rows":len(values), "dimension":spec["dimension"],
                            "norm_min":min(norms),"norm_max":max(norms),
                            "source_url":source.get("source_url"),
                            "acquired_at":source.get("acquired_at")}
        if sources["base"]["rows"] < spec["n_base"]:
            raise ValueError("base prefix larger than source")
        datasets[dataset] = {**spec, "sources":sources}
    Path(args.out).mkdir(parents=True,exist_ok=False)
    seal(Path(args.out)/"inventory.json", {"registry":identity(args.registry),"datasets":datasets})

def audit_queries(args):
    reg = registry(args.registry)
    history = load(args.history_roots)
    roots = [Path(p) for p in history.get("roots",[])]
    records, uncertain, evidence = [], [], []
    known = {}
    for dataset,spec in reg["datasets"].items():
        for key,source in spec["query_sources"].items():
            known[str(Path(source["path"]).resolve())] = (dataset,key,identity(source["path"])["sha256"])
    def walk(value, origin):
        if isinstance(value,list):
            for item in value: walk(item,origin)
        elif isinstance(value,dict):
            source = value.get("query_path") or value.get("queries")
            if isinstance(source,dict): source=source.get("path")
            if isinstance(source,str):
                p=Path(source)
                if not p.is_absolute(): p=origin.parent/p
                entry=known.get(str(p.resolve()))
                if entry:
                    dataset,key,sha=entry
                    if isinstance(value.get("query_ids"),list):
                        ids=value["query_ids"]
                    elif isinstance(value.get("query_count"),int):
                        start=value.get("query_start",0)
                        ids=list(range(start,start+value["query_count"]))
                    else:
                        ids=None
                    if ids is not None and ids and all(isinstance(i,int) and not isinstance(i,bool) and i>=0 for i in ids):
                        records.append({"dataset_id":dataset,"source":key,"source_sha256":sha,
                                        "ids":ids,"status":"used","evidence":str(origin.resolve())})
                    else:
                        uncertain.append({"path":str(origin),"reason":"query source without resolvable IDs"})
            for item in value.values(): walk(item,origin)
    for root in roots:
        if not root.exists():
            uncertain.append({"path":str(root),"reason":"missing history root"});continue
        for path in root.rglob("*.json"):
            if ".git" in path.parts: continue
            try:
                if path.stat().st_size > 64*1024*1024:
                    uncertain.append({"path":str(path),"reason":"oversized JSON requires explicit ledger review"});continue
                walk(load(path),path)
                evidence.append(identity(path))
            except (OSError,ValueError,UnicodeError) as error:
                uncertain.append({"path":str(path),"reason":str(error)})
    # Explicit review records are required for unused ranges. Absence of a hit is NOT evidence.
    reviews = load(history["review_ledger"]) if history.get("review_ledger") else []
    for record in reviews:
        if record.get("status") not in ("used","unused_confirmed","uncertain"):
            raise ValueError("invalid review status")
        dataset,key=record["dataset_id"],record["source"]
        source=reg["datasets"][dataset]["query_sources"][key]
        if record["source_sha256"] != identity(source["path"])["sha256"]:
            raise ValueError("review source hash mismatch")
        record=dict(record)
        if "range" in record:
            start,end=record.pop("range")
            if not 0 <= start < end: raise ValueError("invalid reviewed range")
            record["ids"]=list(range(start,end))
        if not record.get("ids") or any(type(i) is not int or i<0 for i in record["ids"]):
            raise ValueError("invalid reviewed IDs")
        proof=identity(record["evidence"])
        record["evidence_identity"]=proof
        if record["status"]=="unused_confirmed" and not record.get("review_complete",False):
            raise ValueError("unused range requires completed history review")
        records.append(record)
    Path(args.out).mkdir(parents=True,exist_ok=False)
    report={"registry":identity(args.registry),"records":records,"unresolved_history":uncertain,
            "history_files":evidence, "history_spec":identity(args.history_roots),
            "freshness_policy":"explicit_review_required; used/vector_overlap always wins"}
    seal(Path(args.out)/"audit.json",report)
    with (Path(args.out)/"query_usage_ledger.jsonl").open("w",encoding="utf-8") as f:
        for record in records: f.write(json.dumps(record,sort_keys=True)+"\n")

def materialize(args):
    audit=unseal(args.audit)
    verify_file(audit["registry"])
    reg=registry(audit["registry"]["path"])
    spec=reg["datasets"][args.dataset]
    splits=load(args.split_spec)["datasets"][args.dataset]
    if set(splits)!={"dev","select","test"}: raise ValueError("need dev/select/test splits")
    d=spec["dimension"]
    raw_base=vectors(spec["base"],d)
    base=raw_base[:spec["n_base"]]
    base_hashes={vector_hash(v) for v in base}
    source_arrays={key:vectors(source,d) for key,source in spec["query_sources"].items()}
    source_ids={key:identity(source["path"]) for key,source in spec["query_sources"].items()}
    used, fresh, unknown={}, {}, {}
    used_vectors=set()
    for record in audit["records"]:
        if record["dataset_id"] != args.dataset: continue
        key=record["source"]
        if record["source_sha256"] != source_ids[key]["sha256"]: raise ValueError("audit source changed")
        if record.get("evidence_identity"): verify_file(record["evidence_identity"])
        ids=set(record["ids"])
        if not ids or min(ids)<0 or max(ids)>=len(source_arrays[key]): raise ValueError("audit IDs out of range")
        target=used if record["status"]=="used" else fresh if record["status"]=="unused_confirmed" else unknown
        target.setdefault(key,set()).update(ids)
        if record["status"] in ("used","uncertain"):
            used_vectors.update(vector_hash(source_arrays[key][i]) for i in ids)
    rng=np.random.default_rng(args.seed)
    pool, origins, split_ids, taken = [], [], {}, set()
    excluded={"base_overlap":0,"duplicate_query":0,"not_confirmed_unused":0}
    for role in ("dev","select","test"):
        rule=splits[role];key=rule["source"];count=int(rule["count"])
        if count<=0:raise ValueError("split counts must be positive")
        candidates=list(rule.get("candidate_ids",range(len(source_arrays[key]))))
        if len(set(candidates))!=len(candidates) or any(type(i) is not int or i<0 or i>=len(source_arrays[key]) for i in candidates):
            raise ValueError("invalid candidate IDs")
        rng.shuffle(candidates);selected=[]
        for row in candidates:
            vector=source_arrays[key][row];vh=vector_hash(vector)
            if vh in base_hashes:excluded["base_overlap"]+=1;continue
            if vh in taken:excluded["duplicate_query"]+=1;continue
            unused=row in fresh.get(key,set()) and row not in used.get(key,set()) and row not in unknown.get(key,set()) and vh not in used_vectors
            if (role=="test" or not rule.get("allow_seen",False)) and not unused:
                excluded["not_confirmed_unused"]+=1;continue
            selected.append(len(pool));pool.append(np.asarray(vector,dtype="<f4").copy());taken.add(vh)
            origins.append({"query_id":len(pool)-1,"source":key,"source_row":row,
                            "source_sha256":source_ids[key]["sha256"],"vector_sha256":vh,
                            "role":role,"history":"confirmed_unused" if unused else "historically_seen_or_uncertain"})
            if len(selected)==count:break
        if len(selected)!=count:raise ValueError(f"insufficient eligible {role} queries: {len(selected)}/{count}")
        split_ids[role]=selected
    destination=Path(args.out).resolve()
    with atomic_output_dir(destination) as root:
        query=root/"query_pool.fvecs";write_vecs(query,np.asarray(pool))
        # Canonical base is copied only when a conversion/prefix is necessary.
        if spec["base"].get("format","fvecs")=="fvecs" and len(raw_base)==len(base):
            base_entry=identity(spec["base"]["path"])
        else:
            target=root/"base.fvecs";write_vecs(target,base);base_entry=identity(target)
            base_entry["path"]=str(destination/"base.fvecs")
        labels=np.arange(len(base),dtype="<u8") if not spec.get("base_labels") else np.load(spec["base_labels"],allow_pickle=False)
        if labels.ndim!=1 or len(labels)!=len(base) or labels.dtype.kind not in "iu" or np.any(labels<0) or np.any(labels>=2**31) or len(np.unique(labels))!=len(labels):
            raise ValueError("base labels must be unique nonnegative int32-compatible labels")
        np.save(root/"base_labels.npy",labels.astype("<u8"))
        from scripts.edge_estimation.final_study.common import write
        write(root/"query_sources.json",origins)
        def published(path):
            entry=identity(path);entry["path"]=str(destination/path.relative_to(root));return entry
        entries={}
        for role,ids in split_ids.items():
            target=root/f"{role}.ids.txt";write_ids(target,ids);entries[role]={"ids":published(target)}
        warm_count=int(splits["dev"].get("warmup_count",100))
        if not 0 < warm_count < len(split_ids["dev"]):
            raise ValueError("warmup_count must leave nonempty measured dev queries")
        warm=root/"warmup.ids.txt";write_ids(warm,split_ids["dev"][:warm_count])
        manifest={"schema_version":2,"dataset_id":args.dataset,"dimension":d,"n_base":len(base),
                  "metric":"squared_l2","n_queries":len(pool),"base":base_entry,
                  "base_labels":published(root/"base_labels.npy"),"query_pool":published(query),
                  "query_sources":published(root/"query_sources.json"),"warmup_ids":published(warm),
                  "splits":entries,"source_files":source_ids,"audit":identity(args.audit),
                  "split_spec":identity(args.split_spec),"split_seed":args.seed,
                  "test_independence_verified":True,"excluded":excluded,
                  "test_source":splits["test"]["source"],"validated":False}
        seal(root/"dataset_manifest.json",manifest)
        # Adapt to the existing real_data_trace_runner schema.
        write(root/"dataset.json",{"dataset":args.dataset,"distance_kind":"squared_l2_float32",
              "dimension":d,"n_base":len(base),"n_query":len(pool),"base_format":"fvecs",
              "query_format":"fvecs","ground_truth_format":"ivecs","base_path":base_entry["path"],
              "query_path":str(destination/"query_pool.fvecs"),
              "ground_truth_path":str(destination/"full_pool_groundtruth.ivecs"),"ground_truth_k":100,
              "checksums":{"base_sha256":base_entry["sha256"]}})

def merge_topk(old_d,old_i,new_d,new_i,k):
    distances=np.concatenate((old_d,new_d),axis=1)
    indices=np.concatenate((old_i,new_i),axis=1)
    selected=np.lexsort((indices,distances),axis=1)[:,:k]
    return np.take_along_axis(distances,selected,axis=1),np.take_along_axis(indices,selected,axis=1)

def exhaustive(base,query,k,base_block=65536,query_block=64,threads=8,backend="faiss"):
    if k<=0 or k>len(base) or min(base_block,query_block,threads)<=0:raise ValueError("invalid exact GT settings")
    n=len(query);best_d=np.full((n,k),np.inf,dtype=np.float64);best_i=np.full((n,k),-1,dtype=np.int64)
    if backend=="faiss":
        import faiss
        faiss.omp_set_num_threads(threads)
    for lo in range(0,len(base),base_block):
        x=np.ascontiguousarray(base[lo:lo+base_block],dtype=np.float32)
        take=min(k,len(x))
        if backend=="faiss":
            flat=faiss.IndexFlatL2(x.shape[1]);flat.add(x)
        for q0 in range(0,n,query_block):
            q=np.ascontiguousarray(query[q0:q0+query_block],dtype=np.float32)
            if backend=="faiss":D,I=flat.search(q,take)
            else:
                # Float64 direct subtraction, bounded memory, no cancellation-prone norm identity.
                D=np.empty((len(q),take));I=np.empty((len(q),take),dtype=np.int64)
                for row,value in enumerate(q):
                    delta=x.astype(np.float64)-value.astype(np.float64)
                    ds=np.einsum("ij,ij->i",delta,delta)
                    order=np.lexsort((np.arange(len(x)),ds))[:take]
                    D[row]=ds[order];I[row]=order
            target=slice(q0,q0+len(q))
            best_d[target],best_i[target]=merge_topk(best_d[target],best_i[target],D,I+lo,k)
    return best_d,best_i

def exact_gt(args):
    path=Path(args.dataset_manifest);data=unseal(path)
    for key in ("base","query_pool","base_labels"):verify_file(data[key])
    spec=data["splits"][args.split];verify_file(spec["ids"])
    ids=read_ids(spec["ids"]["path"],data["n_queries"])
    base=read_vecs(data["base"]["path"],dimension=data["dimension"])
    pool=read_vecs(data["query_pool"]["path"],dimension=data["dimension"])
    labels=np.load(data["base_labels"]["path"],allow_pickle=False)
    D,I=exhaustive(base,pool[ids],args.k,args.base_block,args.query_block,args.threads,args.backend)
    root=path.parent
    gt=root/f"{args.split}.ivecs";dist=root/f"{args.split}.distances.npy";rows=root/f"{args.split}.gt_rows.json"
    if any(p.exists() for p in (gt,dist,rows)):raise ValueError("GT output already exists; use a new dataset version")
    write_vecs(gt,labels[I].astype("<i4"),"i");np.save(dist,D)
    from scripts.edge_estimation.final_study.common import write
    write(rows,{str(q):i for i,q in enumerate(ids)})
    spec.update(gt=identity(gt),gt_distances=identity(dist),gt_rows=identity(rows),
                gt_parameters={"backend":args.backend,"k":args.k,"base_block":args.base_block,
                               "query_block":args.query_block,"threads":args.threads,
                               "tie_break":"distance_then_base_row; boundary_ties_require_validation"})
    seal(path,data)

def validate(args):
    path=Path(args.dataset_manifest);data=unseal(path)
    for key in ("base","query_pool","base_labels","query_sources","warmup_ids","audit","split_spec"):
        verify_file(data[key])
    if not data.get("test_independence_verified"):raise ValueError("query independence not verified")
    base=read_vecs(data["base"]["path"],dimension=data["dimension"])
    pool=read_vecs(data["query_pool"]["path"],dimension=data["dimension"])
    labels=np.load(data["base_labels"]["path"],allow_pickle=False);allowed=set(map(int,labels))
    seen=set();checks={}
    rng=np.random.default_rng(args.seed)
    for split,spec in data["splits"].items():
        for key in ("ids","gt","gt_rows","gt_distances"):verify_file(spec[key])
        ids=read_ids(spec["ids"]["path"],len(pool))
        if seen.intersection(ids):raise ValueError("query split overlap")
        seen.update(ids)
        truth=read_vecs(spec["gt"]["path"],"i")
        distances=np.load(spec["gt_distances"]["path"],mmap_mode="r",allow_pickle=False)
        if distances.shape!=truth.shape or not np.isfinite(distances).all() or np.any(distances<0) or np.any(distances[:,1:]<distances[:,:-1]):
            raise ValueError("invalid or unsorted exact GT distance matrix")
        rows=load(spec["gt_rows"]["path"])
        if len(rows)!=len(ids) or set(rows)!=set(map(str,ids)) or set(rows.values())!=set(range(len(ids))) or len(truth)!=len(ids):
            raise ValueError("compact GT mapping mismatch")
        for row in truth:
            if len(set(map(int,row)))!=len(row) or not set(map(int,row)).issubset(allowed):
                raise ValueError("invalid GT label row")
        chosen=rng.choice(ids,min(args.float64_check_queries,len(ids)),replace=False)
        D,I=exhaustive(base,pool[chosen],min(100,len(base)),min(4096,args.base_block),1,1,"float64")
        mismatches=[]
        for i,q in enumerate(chosen):
            k=min(10,truth.shape[1]);actual=list(map(int,truth[rows[str(int(q))],:k]))
            expected=list(map(int,labels[I[i,:k]]))
            if actual!=expected:
                mismatches.append(int(q))
        checks[split]={"checked_queries":list(map(int,chosen)),"top10_order_mismatches":mismatches}
        if mismatches:
            raise ValueError(f"float64 GT mismatch in {split}: {mismatches[:10]}; regenerate affected GT with --backend float64")
    warm=read_ids(data["warmup_ids"]["path"],len(pool))
    if not set(warm).issubset(set(read_ids(data["splits"]["dev"]["ids"]["path"]))):
        raise ValueError("warmup must be from dev")
    from scripts.edge_estimation.final_study.common import write
    write(path.parent/"gt_validation.json",checks)
    # Existing graph builder expects a full-pool GT; assemble it with the canonical mapping.
    width=min(read_vecs(s["gt"]["path"],"i").shape[1] for s in data["splits"].values())
    full=np.empty((len(pool),width),dtype="<i4")
    for spec in data["splits"].values():
        gt=read_vecs(spec["gt"]["path"],"i");mapping=load(spec["gt_rows"]["path"])
        for query,row in mapping.items():full[int(query)]=gt[row,:width]
    if seen!=set(range(len(pool))):raise ValueError("pool contains unassigned query rows")
    write_vecs(path.parent/"full_pool_groundtruth.ivecs",full,"i")
    data["validated"]=True;data["gt_validation"]=identity(path.parent/"gt_validation.json")
    seal(path.parent/"dataset_contract.json",data)
    print(json.dumps({"valid":True,"contract":str(path.parent/"dataset_contract.json")}))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="stage",required=True)
    for stage in ("inventory","audit-queries"):
        x=sub.add_parser(stage);x.add_argument("--registry",required=True);x.add_argument("--out",required=True)
        if stage=="audit-queries":x.add_argument("--history-roots",required=True)
    x=sub.add_parser("materialize")
    for key in ("dataset","audit","split-spec","out"):x.add_argument("--"+key,required=True)
    x.add_argument("--seed",type=int,default=20261008)
    x=sub.add_parser("exact-gt");x.add_argument("--dataset-manifest",required=True)
    x.add_argument("--split",choices=("dev","select","test"),required=True)
    x.add_argument("--k",type=int,default=100);x.add_argument("--query-block",type=int,default=64)
    x.add_argument("--base-block",type=int,default=65536);x.add_argument("--threads",type=int,default=8)
    x.add_argument("--backend",choices=("faiss","float64"),default="faiss")
    x=sub.add_parser("validate");x.add_argument("--dataset-manifest",required=True)
    x.add_argument("--float64-check-queries",type=int,default=100);x.add_argument("--seed",type=int,default=20261008)
    x.add_argument("--base-block",type=int,default=4096)
    args=p.parse_args()
    {"inventory":inventory,"audit-queries":audit_queries,"materialize":materialize,
     "exact-gt":exact_gt,"validate":validate}[args.stage](args)
if __name__=="__main__":cli(main)
