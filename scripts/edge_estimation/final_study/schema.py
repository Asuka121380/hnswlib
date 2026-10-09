from __future__ import annotations
import math
from .common import load, name, positive

METHODS = ("hnsw", "opq", "ivf_opq")
PHASES = ("dev", "batch", "select", "test", "diagnostic", "mechanism")
DEFAULT_EF = {
    "gist1m": [200,350,400,450,600,700,900,1100,1300],
    "sift1m": [25,50,75,100,150,200,300,400,600],
    "deep1m_yandex10m_prefix_v1": [50,100,150,200,300,400,600,800,1200],
}

def protocol(path):
    p = load(path)
    if p.get("schema_version") != 2:
        raise ValueError("final study requires schema_version=2")
    name(p["study_id"])
    if p.get("methods") != list(METHODS):
        raise ValueError("core protocol must declare hnsw/opq/ivf_opq")
    b = p["build"]
    if b.get("native") is not False or b.get("type") != "Release" or b.get("optimization") != "O3" or b.get("strict_fp") is not True:
        raise ValueError("expected common portable Release/O3/strict-FP build")
    if p["search"].get("threads") != 1 or p["search"].get("blas_threads") != 1:
        raise ValueError("search and BLAS must both be single threaded")
    positive(p["search"]["k"], "k")
    if p["measurement"].get("qps_denominator") != "full_service_wall":
        raise ValueError("invalid QPS denominator")
    for target in p["selection"]["recall_targets"]:
        if not isinstance(target, (int,float)) or not 0 < target <= 1:
            raise ValueError("invalid recall target")
    for key in ("allocations", "blocks_per_allocation"):
        positive(p["measurement"][key], key)
    positive(p["measurement"]["result_buffer_cap_bytes"], "result buffer cap")
    if not math.isfinite(p["measurement"]["minimum_case_seconds"]) or p["measurement"]["minimum_case_seconds"] <= 0:
        raise ValueError("positive finite minimum case duration required")
    training=p["training"];trainer=training["trainer"]
    for key in ("m","nbits","coarse_centers","coarse_iterations","sample_count"):
        positive(training[key],key)
    for key in ("seed","threads","iterations","outer_iterations","initial_pq_iterations","redos","max_train_points","max_points_per_centroid","encode_batch_size"):
        positive(trainer[key],key)
    if trainer["provider"]!="faiss" or trainer["threads"]!=1 or training["nbits"]>8:
        raise ValueError("expected single-thread Faiss product quantizers")
    if trainer["max_train_points"]<training["sample_count"] or trainer["max_points_per_centroid"]*(1<<training["nbits"])<training["sample_count"]:
        raise ValueError("trainer limits would subsample the shared training edges")
    if not p["selection"]["recall_targets"] or len(set(p["selection"]["recall_targets"]))!=len(p["selection"]["recall_targets"]):
        raise ValueError("recall targets must be nonempty and unique")
    if p["selection"].get("batch_scope")!="per_dataset_shared":
        raise ValueError("this protocol requires a preregistered per-dataset, method-shared batch strategy")
    positive(p["selection"]["scratch_budget_bytes"],"scratch budget")
    if not p["development"]["beta_grid"] or any(type(b) not in (int,float) or not math.isfinite(b) or b<=0 for b in p["development"]["beta_grid"]):
        raise ValueError("invalid shared beta grid")
    return p

def case(value, k=10):
    c = dict(value)
    if c.get("method") not in METHODS:
        raise ValueError("unsupported study method")
    positive(c["ef_search"], "ef_search")
    if c["ef_search"] < k:
        raise ValueError("ef_search must be at least k")
    beta = c.get("beta")
    if c["method"] == "hnsw":
        if beta is not None:
            raise ValueError("baseline must not have beta")
    elif isinstance(beta,bool) or not isinstance(beta,(int,float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("invalid beta")
    for field in ("prepare_window", "compute_chunk"):
        if c.get(field) != "all":
            positive(c.get(field), field)
    if c["prepare_window"] != "all" and c["compute_chunk"] != "all" and c["compute_chunk"] > c["prepare_window"]:
        raise ValueError("compute chunk exceeds window")
    positive(c.get("inner_repeats",1), "inner_repeats")
    return c

def unique_cases(values, k=10):
    from .common import digest
    result, seen = [], set()
    for value in values:
        c = case(value,k)
        identity = {key:c.get(key) for key in ("method","ef_search","beta","prepare_window","compute_chunk","inner_repeats")}
        key = digest(identity)[:20]
        if key in seen:
            continue
        seen.add(key)
        result.append({**c, "case_config_id":key})
    if not result:
        raise ValueError("empty case list")
    return result
