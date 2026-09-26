from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ScoredEvent:
    query_id: int
    threshold_valid: bool
    threshold: float
    exact_squared_distance: float
    estimated_squared_distance: float | None
    status: str
    heap_full: bool = True


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _error_summary(values: Sequence[float]) -> Mapping[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0,
                "p99": 0.0, "max": 0.0}
    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower, upper = int(math.floor(position)), int(math.ceil(position))
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return {"count": len(ordered), "p50": percentile(0.50),
            "p90": percentile(0.90), "p95": percentile(0.95),
            "p99": percentile(0.99), "max": ordered[-1]}


def summarize(events: Sequence[ScoredEvent], alpha: float) -> Mapping[str, object]:
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    status_counts = Counter(event.status for event in events)
    decision_set = [
        event for event in events
        if event.heap_full and event.threshold_valid
        and math.isfinite(event.threshold) and event.threshold >= 0
        and math.isfinite(event.exact_squared_distance)
    ]
    tp = fp = fn = tn = valid = 0
    per_query: dict[int, Counter[str]] = {}
    per_query_errors: dict[int, tuple[list[float], list[float]]] = {}
    overestimate: list[float] = []
    underestimate: list[float] = []
    for event in decision_set:
        is_valid = (event.status == "valid" and
                    event.estimated_squared_distance is not None and
                    math.isfinite(event.estimated_squared_distance))
        valid += int(is_valid)
        if is_valid:
            signed_error = float(event.estimated_squared_distance) - event.exact_squared_distance
            over, under = max(0.0, signed_error), max(0.0, -signed_error)
            overestimate.append(over); underestimate.append(under)
            query_over, query_under = per_query_errors.setdefault(
                event.query_id, ([], []))
            query_over.append(over); query_under.append(under)
        prune = bool(is_valid and
                     event.estimated_squared_distance > alpha * event.threshold)
        far = event.exact_squared_distance > event.threshold
        outcome = "tp" if prune and far else "fp" if prune else "fn" if far else "tn"
        if outcome == "tp": tp += 1
        elif outcome == "fp": fp += 1
        elif outcome == "fn": fn += 1
        else: tn += 1
        per_query.setdefault(event.query_id, Counter())[outcome] += 1
    total = len(decision_set)
    far_count = tp + fn
    non_far_count = fp + tn
    return {
        "schema_version": 1,
        "alpha": alpha,
        "event_count_u": len(events),
        "decision_count_s": total,
        "valid_estimate_count": valid,
        "fallback_count": total - valid,
        "status_counts": dict(sorted(status_counts.items())),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "metrics": {
            "correct_prune_rate": _ratio(tp, total),
            "prunable_coverage": _ratio(tp, far_count),
            "global_false_prune_rate": _ratio(fp, total),
            "conditional_false_prune_rate": _ratio(fp, non_far_count),
            "prune_precision": _ratio(tp, tp + fp),
            "total_prune_rate": _ratio(tp + fp, total),
            "valid_estimate_coverage": _ratio(valid, total),
        },
        "error_percentiles": {"overestimate": _error_summary(overestimate),
                              "underestimate": _error_summary(underestimate)},
        "per_query": {
            str(key): {"counts": dict(value),
                       "overestimate": _error_summary(
                           per_query_errors.get(key, ([], []))[0]),
                       "underestimate": _error_summary(
                           per_query_errors.get(key, ([], []))[1])}
            for key, value in sorted(per_query.items())
        },
    }
