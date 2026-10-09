"""Sample one audited set of nonzero graph edges for both quantizers."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .common import cli, seal, unseal, identity, verify_file
from scripts.edge_estimation.ivf_train import context
from scripts.edge_estimation.train_encode import edge_geometry

def read_sample(path, assets, catalog, dimension):
    doc=unseal(path)
    _,_,offsets,targets,cat,hashes=context(assets,catalog,dimension)
    if doc["assets"]!=hashes or doc["catalog_identity"]!=cat["catalog_identity"]:
        raise ValueError("training sample graph binding mismatch")
    ids=np.fromfile(verify_file(doc["ids"]),dtype="<u8")
    if len(ids)!=doc["count"] or len(ids)==0 or len(np.unique(ids))!=len(ids) or ids.max()>=len(targets):
        raise ValueError("invalid shared edge IDs")
    return ids.astype(np.int64)

def sample(assets,catalog,dimension,count,seed,out):
    if count<=0:raise ValueError("positive sample count required")
    out=Path(out)
    if out.exists():raise ValueError("sample output already exists")
    base,mapping,offsets,targets,cat,hashes=context(assets,catalog,dimension)
    # Random-priority reservoir over valid edges: O(count) retained memory, unbiased,
    # including when duplicate base vectors make many graph edges zero length.
    rng=np.random.default_rng(seed); ids=np.empty(0,dtype=np.int64); priorities=np.empty(0)
    for first in range(0,len(targets),4096):
        chunk=np.arange(first,min(first+4096,len(targets)))
        _,_,lengths,_=edge_geometry(base,mapping,offsets,targets,chunk)
        valid=np.isfinite(lengths)&(lengths>0)
        chunk=chunk[valid]; scores=rng.random(len(chunk))
        ids=np.concatenate((ids,chunk));priorities=np.concatenate((priorities,scores))
        if len(ids)>count:
            take=np.argpartition(priorities,count-1)[:count]
            ids,priorities=ids[take],priorities[take]
    if len(ids)<count:raise ValueError("not enough valid graph edges for requested sample")
    out.mkdir(parents=True); ids=np.sort(ids); ids.astype("<u8").tofile(out/"ids.u64le")
    seal(out/"sample.json",{"schema_version":2,"seed":seed,"count":count,"dimension":dimension,
        "assets":hashes,"catalog_identity":cat["catalog_identity"],"ids":identity(out/"ids.u64le"),
        "algorithm":"uniform_random_priority_valid_edges"})
    return out/"sample.json"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ("assets","catalog","out"):p.add_argument("--"+x,required=True)
    p.add_argument("--dimension",required=True,type=int);p.add_argument("--count",default=100000,type=int)
    p.add_argument("--seed",default=42,type=int);a=p.parse_args()
    print(sample(a.assets,a.catalog,a.dimension,a.count,a.seed,a.out))
if __name__=="__main__":cli(main)
