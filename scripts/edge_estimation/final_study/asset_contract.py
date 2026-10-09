"""Bind quantizers to the exact graph, base rows, external labels and training sample."""
from __future__ import annotations
import argparse,struct
from pathlib import Path
import numpy as np
from .common import cli,identity,seal,unseal,load,verify_file
from scripts.edge_estimation.ivf_train import context,verify

def native(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        k,v=line.split("=",1)
        if k in result:raise ValueError("duplicate native configuration key")
        result[k]=v
    return result

def bind(dataset_path,assets_path,catalog,opq,cr,sample_path,out):
    data=unseal(dataset_path)
    if not data.get("validated") or not data.get("test_independence_verified"):
        raise ValueError("validated dataset with audited independent test queries required")
    assets=load(assets_path);base,mapping,offsets,targets,cat,hashes=context(assets_path,catalog,data["dimension"])
    if hashes["base"]!=data["base"]["sha256"]:raise ValueError("dataset and graph base differ")
    labels=np.load(verify_file(data["base_labels"]),allow_pickle=False)
    header=struct.Struct("<6QiI3QdQ")
    with Path(assets["index"]).open("rb") as f:fields=header.unpack(f.read(header.size))
    raw=np.memmap(assets["index"],dtype="u1",mode="r")
    graph_labels=np.ndarray((len(mapping),),dtype="<u8",buffer=raw,offset=header.size+fields[4],strides=(fields[3],))
    if not np.array_equal(graph_labels,labels[mapping.astype(np.int64)]):raise ValueError("graph external labels differ from GT label space")
    shared=unseal(sample_path);verify_file(shared["ids"])
    if shared["assets"]!=hashes:raise ValueError("shared sample graph mismatch")
    training=load(Path(opq)/"training.json");cm=verify(cr);coarse=cm["coarse_manifest"]
    if training.get("bound_assets")!=hashes or cm["assets"]!=hashes:raise ValueError("quantizers lack exact graph binding")
    if training["sample_ids_sha256"]!=shared["ids"]["sha256"] or coarse["sample_ids_sha256"]!=shared["ids"]["sha256"]:
        raise ValueError("OPQ and CR-OPQ did not use identical edge IDs")
    for key in ("provider","seed","threads","iterations","outer_iterations","initial_pq_iterations","redos","max_train_points","max_points_per_centroid"):
        if key not in training["parameters"] or training["parameters"][key]!=cm["trainer"].get(key):
            raise ValueError("unequal effective training parameter: "+key)
    opm=load(Path(opq)/"manifest.json")
    if (opm["codec"]["m"],opm["codec"]["nbits"])!=(cm["m"],cm["nbits"]):raise ValueError("unequal codec shapes")
    artifacts={}
    for method,directory in (("opq",opq),("ivf_opq",cr)):
        directory=Path(directory).resolve();cfg=native(directory/"native.cfg")
        if cfg.get("index_sha256")!=hashes["index"]:raise ValueError("native graph binding missing")
        artifacts[method]={"path":str(directory),"files":[identity(p) for p in sorted(directory.rglob("*")) if p.is_file()]}
    if Path(out).exists():raise ValueError("asset contract is immutable; use a new output")
    seal(out,{"schema_version":2,"dataset_id":data["dataset_id"],"dataset":identity(dataset_path),
        "assets_source":identity(assets_path),"index":identity(assets["index"]),"base":data["base"],
        "mapping":identity(assets["internal_to_label"]),"base_labels":data["base_labels"],
        "catalog_files":[identity(p) for p in sorted(Path(catalog).rglob("*")) if p.is_file()],
        "catalog_identity":cat["catalog_identity"],"edge_count":len(targets),
        "shared_sample":identity(sample_path),"artifacts":artifacts,
        "training_spec":{"m":cm["m"],"nbits":cm["nbits"],"coarse_centers":cm["nlist"],
            "coarse_iterations":coarse["iterations"],"sample_count":shared["count"],
            "trainer":training["parameters"]}})
def verify_assets(path):
    doc=unseal(path)
    for key in ("dataset","assets_source","index","base","mapping","base_labels","shared_sample"):verify_file(doc[key])
    for f in doc["catalog_files"]:verify_file(f)
    for art in doc["artifacts"].values():
        for f in art["files"]:verify_file(f)
    return doc
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("dataset","assets","catalog","opq","cr-opq","sample","out"):p.add_argument("--"+key,required=True)
    a=p.parse_args();bind(a.dataset,a.assets,a.catalog,a.opq,a.cr_opq,a.sample,a.out)
if __name__=="__main__":cli(main)
