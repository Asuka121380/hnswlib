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
    for event in decision_set:
        is_valid = (event.status == "valid" and
                    event.estimated_squared_distance is not None and
                    math.isfinite(event.estimated_squared_distance))
        valid += int(is_valid)
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
        "per_query": {str(key): dict(value) for key, value in sorted(per_query.items())},
    }
