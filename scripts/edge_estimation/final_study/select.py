"""Select one shared batching strategy using select queries only."""
from __future__ import annotations
import argparse, math
from pathlib import Path
from .common import cli,seal,identity
from .schema import protocol
from .results import rows,aggregate

def choose(a):
    raw,manifests=rows(a.runs,"batch")
    bindings=manifests[0]["bindings"]
    if any(m["bindings"]!=bindings for m in manifests):raise ValueError("mixed batch inputs")
    p=protocol(bindings["protocol"]["path"]);points=aggregate(raw)
    if any(x["count"]<3 for x in points) and not all(m["local_smoke"] for m in manifests):
        raise ValueError("at least three complete blocks required")
    if any(x["qps_cv"]>p["selection"]["noise_cv_limit"] and x["count"]<9 for x in points):
        raise ValueError("batch timing is noisy; gather nine blocks before selecting")
    candidates=[]
    for window,chunk in ((128,128),("all",128),("all","all")):
        scores=[];valid=True
        for method in ("opq","ivf_opq"):
            pool=[x for x in points if x["case"]["method"]==method]
            reference=next(x for x in pool if x["case"]["prepare_window"]==128 and x["case"]["compute_chunk"]==128)
            chosen=[x for x in pool if x["case"]["prepare_window"]==window and x["case"]["compute_chunk"]==chunk]
            if len(chosen)!=1:raise ValueError("incomplete batch design")
            point=chosen[0]
            valid &= point["per_query"]==reference["per_query"]
            valid &= point["scratch_bytes_max"]<=p["selection"]["scratch_budget_bytes"]
            # A stable >2% loss for either method is not traded for the other's gain.
            valid &= not (point["qps_service"]<reference["qps_service"]*.98 and
                          max(point["qps_cv"],reference["qps_cv"])<=p["selection"]["noise_cv_limit"])
            scores.append(point["qps_service"]/reference["qps_service"])
        candidates.append({"prepare_window":window,"compute_chunk":chunk,"valid":bool(valid),
                           "geometric_relative_qps":math.prod(scores)**.5})
    usable=[x for x in candidates if x["valid"]]
    if not usable:raise ValueError("no batch strategy preserves per-query recall")
    best=max(usable,key=lambda x:x["geometric_relative_qps"])
    # Within 2%, prefer the bounded LUT window.
    bounded=usable[0]
    if bounded["prepare_window"]==128 and bounded["geometric_relative_qps"]>=best["geometric_relative_qps"]*.98:best=bounded
    if Path(a.out).exists():raise ValueError("immutable batch choice already exists")
    seal(a.out,{"schema_version":2,"bindings":bindings,**best,"candidates":candidates,
       "local_smoke":any(m["local_smoke"] for m in manifests),
       "evidence":[identity(Path(x)/"complete.json" if Path(x).is_dir() else x) for x in a.runs]})
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--runs",nargs="+",required=True)
    p.add_argument("--out",required=True);choose(p.parse_args())
if __name__=="__main__":cli(main)
