from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _alpha_list(value: object, name: str, *, minimum: float = 0.0) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = [_number(item, f"{name} item", minimum=minimum) for item in value]
    if result != sorted(result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def validate_quality_policy(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("quality policy must be a schema_version=1 object")
    if not isinstance(value.get("protocol_id"), str) or not value["protocol_id"]:
        raise ValueError("quality policy protocol_id must be non-empty")
    if value.get("policy_kind") != "scaled_threshold":
        raise ValueError("only scaled_threshold quality policies are supported")
    diagnostic = _alpha_list(value.get("diagnostic_alphas"), "diagnostic_alphas")
    candidates = _alpha_list(
        value.get("candidate_alphas"), "candidate_alphas", minimum=1.0)
    if set(diagnostic).intersection(candidates):
        raise ValueError("diagnostic and candidate alphas must be disjoint")
    gates = value.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ValueError("quality policy gates must be a non-empty object")
    for name, raw_gate in gates.items():
        if not isinstance(name, str) or not isinstance(raw_gate, Mapping):
            raise ValueError("quality policy gates must be named objects")
        for key in ("max_global_false_prune_rate",
                    "max_p95_query_false_prune_rate"):
            rate = _number(raw_gate.get(key), f"gates.{name}.{key}")
            if rate > 1.0:
                raise ValueError(f"gates.{name}.{key} must be <= 1")
    fine = value.get("fine_sweep")
    if not isinstance(fine, Mapping):
        raise ValueError("quality policy fine_sweep must be an object")
    _number(fine.get("half_width"), "fine_sweep.half_width")
    if _number(fine.get("step"), "fine_sweep.step") <= 0.0:
        raise ValueError("fine_sweep.step must be positive")
    _number(fine.get("minimum_candidate_alpha"),
            "fine_sweep.minimum_candidate_alpha", minimum=1.0)
    _alpha_list(fine.get("extension_alphas"), "fine_sweep.extension_alphas",
                minimum=1.0)


def sweep_alphas(policy: Mapping[str, Any]) -> list[float]:
    validate_quality_policy(policy)
    return sorted(float(value) for value in (
        list(policy["diagnostic_alphas"]) + list(policy["candidate_alphas"])))


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze_native_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    required = ("alpha", "decision_count_s", "valid_estimate_count",
                "fallback_count", "tp", "fp", "fn", "tn", "per_query")
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError("native quality summary is missing: " + ", ".join(missing))
    alpha = _number(summary["alpha"], "summary.alpha")
    counts = {key: int(summary[key]) for key in
              ("decision_count_s", "valid_estimate_count", "fallback_count",
               "tp", "fp", "fn", "tn")}
    if any(value < 0 for value in counts.values()):
        raise ValueError("native quality counters must be non-negative")
    decision_count = counts["decision_count_s"]
    if counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"] != decision_count:
        raise ValueError("native confusion counts do not cover the decision set")
    if counts["valid_estimate_count"] + counts["fallback_count"] != decision_count:
        raise ValueError("valid and fallback counts do not cover the decision set")
    raw_per_query = summary["per_query"]
    if not isinstance(raw_per_query, list):
        raise ValueError("native per_query must be a list")
    per_query_false_prune_rates: list[float] = []
    for item in raw_per_query:
        if not isinstance(item, Mapping) or not isinstance(item.get("counts"), Mapping):
            raise ValueError("native per_query entry is invalid")
        query_counts = item["counts"]
        query_decisions = int(query_counts.get("decision_count_s", 0))
        query_fp = int(query_counts.get("fp", 0))
        if query_decisions < 0 or query_fp < 0 or query_fp > query_decisions:
            raise ValueError("native per-query counters are invalid")
        if query_decisions:
            per_query_false_prune_rates.append(query_fp / query_decisions)
    tp, fp = counts["tp"], counts["fp"]
    fn, tn = counts["fn"], counts["tn"]
    metrics = {
        "correct_prune_rate": _ratio(tp, decision_count),
        "global_false_prune_rate": _ratio(fp, decision_count),
        "conditional_false_prune_rate": _ratio(fp, fp + tn),
        "prunable_coverage": _ratio(tp, tp + fn),
        "prune_precision": _ratio(tp, tp + fp),
        "total_prune_rate": _ratio(tp + fp, decision_count),
        "valid_estimate_coverage": _ratio(
            counts["valid_estimate_count"], decision_count),
        "p95_query_false_prune_rate": _percentile(
            per_query_false_prune_rates, 0.95),
        "queries_with_decisions": len(per_query_false_prune_rates),
    }
    return {"alpha": alpha, "counts": counts, "metrics": metrics}


def _fine_grid(anchor: float, policy: Mapping[str, Any]) -> list[float]:
    fine = policy["fine_sweep"]
    width = float(fine["half_width"])
    step = float(fine["step"])
    minimum = float(fine["minimum_candidate_alpha"])
    first = max(minimum, anchor - width)
    last = anchor + width
    count = int(math.floor((last - first) / step + 1e-9))
    result = [round(first + index * step, 12) for index in range(count + 1)]
    if not result or result[-1] < last - 1e-9:
        result.append(round(last, 12))
    return sorted(set(result))


def select_operating_points(
        analyses: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any]) -> dict[str, Any]:
    validate_quality_policy(policy)
    candidates = {float(value) for value in policy["candidate_alphas"]}
    by_alpha = {float(item["alpha"]): item for item in analyses}
    missing = sorted(candidates - set(by_alpha))
    if missing:
        raise ValueError(f"quality sweep is missing candidate alphas: {missing}")
    selections: dict[str, Any] = {}
    for gate_name, gate in policy["gates"].items():
        eligible: list[Mapping[str, Any]] = []
        for alpha in sorted(candidates):
            item = by_alpha[alpha]
            metrics = item["metrics"]
            global_fp = metrics["global_false_prune_rate"]
            query_p95 = metrics["p95_query_false_prune_rate"]
            if (global_fp is not None and query_p95 is not None and
                    global_fp <= float(gate["max_global_false_prune_rate"]) and
                    query_p95 <= float(gate["max_p95_query_false_prune_rate"])):
                eligible.append(item)
        if not eligible:
            selections[gate_name] = {
                "status": "no_candidate_meets_gate",
                "extension_alphas": list(policy["fine_sweep"]["extension_alphas"]),
            }
            continue
        def rank(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
            metrics = item["metrics"]
            return (
                float(metrics["correct_prune_rate"] or 0.0),
                -float(metrics["global_false_prune_rate"] or 0.0),
                float(metrics["prune_precision"] or 0.0),
                -float(item["alpha"]),
            )
        chosen = max(eligible, key=rank)
        selections[gate_name] = {
            "status": "selected",
            "alpha": float(chosen["alpha"]),
            "metrics": chosen["metrics"],
            "fine_sweep_alphas": _fine_grid(float(chosen["alpha"]), policy),
        }
    return {
        "schema_version": 1,
        "protocol_id": policy["protocol_id"],
        "objective": policy["selection"]["objective"],
        "selections": selections,
    }
