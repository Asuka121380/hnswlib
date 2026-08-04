#!/usr/bin/env python3
"""Phase-1 high-precision spherical-cap evaluator.

The C++ runner exports sampled raw geometry only.  This program computes the
cap support and lower bound independently with Decimal arithmetic and checks
the result with a separate two-dimensional candidate maximisation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


DECIMAL_PRECISION = 80
CAP_OUTPUT_SCHEMA_VERSION = 1
REQUIRED_COLUMNS = {
    "cap_input_schema_version", "run_id", "query_id", "current_node_id",
    "candidate_id", "edge_length", "direction_error", "threshold",
    "current_lb", "exact_squared_distance", "current_would_prune",
    "oracle_would_prune", "reconstruction_norm", "x_norm", "x_dot_r",
    "true_edge_norm", "x_dot_true_direction", "actual_direction_error",
    "certificate_slack", "diagnostic_valid", "raw_would_prune",
}


def decimal_value(row: Mapping[str, str], name: str) -> Decimal:
    try:
        value = Decimal(row[name])
    except (KeyError, InvalidOperation) as exc:
        raise ValueError(f"invalid decimal field {name!r}") from exc
    if not value.is_finite():
        raise ValueError(f"non-finite decimal field {name!r}")
    return value


def bool_value(row: Mapping[str, str], name: str) -> bool:
    value = row[name].strip().lower()
    if value in {"1", "true"}:
        return True
    if value in {"0", "false"}:
        return False
    raise ValueError(f"invalid boolean field {name!r}: {row[name]!r}")


def dec_sqrt(value: Decimal) -> Decimal:
    if value < 0:
        raise ValueError("negative square root")
    return value.sqrt()


def cap_geometry(
    reconstruction_norm: Decimal,
    direction_error: Decimal,
) -> Tuple[str, Decimal | None, bool]:
    s = reconstruction_norm
    epsilon = direction_error
    if s < 0 or epsilon < 0:
        return "numeric_failure", None, False
    if s == 0:
        return "s_zero", None, epsilon >= 1
    if epsilon < abs(s - 1):
        return "empty", None, False
    if epsilon >= s + 1:
        return "full_sphere", Decimal(-1), True
    kappa = (1 + s * s - epsilon * epsilon) / (2 * s)
    if kappa > 1 or kappa < -1:
        return "numeric_failure", kappa, False
    if epsilon == abs(s - 1):
        return "singleton", kappa, True
    return "proper", kappa, True


def closed_form_support(
    kind: str,
    kappa: Decimal | None,
    x_norm: Decimal,
    x_dot_r: Decimal,
    reconstruction_norm: Decimal,
) -> Tuple[Decimal, str, Decimal | None]:
    if x_norm == 0:
        return Decimal(0), "inside", Decimal(0)
    if kind in {"full_sphere", "s_zero"}:
        return x_norm, "fallback", None
    if kind not in {"proper", "singleton"} or kappa is None:
        raise ValueError("cap does not have a valid support")
    t = x_dot_r / reconstruction_norm
    if t >= x_norm * kappa:
        return x_norm, "inside", t
    perpendicular_squared = x_norm * x_norm - t * t
    cap_squared = 1 - kappa * kappa
    scale = max(Decimal(1), x_norm * x_norm)
    tolerance = Decimal(64) * Decimal(10) ** (-(DECIMAL_PRECISION - 8)) * scale
    if perpendicular_squared < 0 and perpendicular_squared >= -tolerance:
        perpendicular_squared = Decimal(0)
    if cap_squared < 0 and cap_squared >= -tolerance:
        cap_squared = Decimal(0)
    support = (
        t * kappa
        + dec_sqrt(perpendicular_squared) * dec_sqrt(cap_squared)
    )
    return support, "boundary", t


def span_candidate_support(
    kind: str,
    kappa: Decimal | None,
    x_norm: Decimal,
    t: Decimal | None,
) -> Decimal:
    """Independent 2D reduction: evaluate all feasible stationary/end points."""
    if x_norm == 0:
        return Decimal(0)
    if kind in {"full_sphere", "s_zero"}:
        return x_norm
    if kappa is None or t is None:
        raise ValueError("2D support requires a proper cap")
    perpendicular = dec_sqrt(max(Decimal(0), x_norm * x_norm - t * t))

    def objective(z: Decimal) -> Decimal:
        return t * z + perpendicular * dec_sqrt(max(Decimal(0), 1 - z * z))

    candidates = [objective(kappa), objective(Decimal(1))]
    stationary = t / x_norm
    if kappa <= stationary <= 1:
        candidates.append(objective(stationary))
    return max(candidates)


def evaluate_row(
    row: Mapping[str, str],
    comparison_tolerance: Decimal,
    oracle_tolerance: Decimal,
    certificate_tolerance: Decimal,
) -> Dict[str, str]:
    output = dict(row)
    output["cap_output_schema_version"] = str(CAP_OUTPUT_SCHEMA_VERSION)
    if not bool_value(row, "diagnostic_valid"):
        output.update({
            "kappa": "", "cap_kind": "numeric_failure",
            "cap_branch": "fallback", "x_dot_r_hat": "",
            "cap_half_angle_radians": "",
            "cap_support": "", "span_support": "",
            "current_ball_support": "", "current_ball_lb": "",
            "cap_lb_high_precision": "", "cap_would_prune": "0",
            "cap_gain": "", "cap_safety_slack": "",
            "gap_recovery": "", "support_contraction": "",
            "certificate_failure": "0", "support_oracle_mismatch": "0",
            "true_direction_support_violation": "0",
            "cap_lower_bound_violation": "0", "cap_below_current": "0",
            "cap_false_prune": "0",
            "cap_valid": "0",
        })
        return output

    ell = decimal_value(row, "edge_length")
    epsilon = decimal_value(row, "direction_error")
    s = decimal_value(row, "reconstruction_norm")
    n = decimal_value(row, "x_norm")
    x_dot_r = decimal_value(row, "x_dot_r")
    threshold = decimal_value(row, "threshold")
    current_lb = decimal_value(row, "current_lb")
    exact = decimal_value(row, "exact_squared_distance")
    actual_error = decimal_value(row, "actual_direction_error")
    true_support = decimal_value(row, "x_dot_true_direction")

    kind, kappa, geometry_valid = cap_geometry(s, epsilon)
    certificate_failure = actual_error > epsilon + certificate_tolerance
    if not geometry_valid:
        output.update({
            "kappa": "" if kappa is None else str(kappa),
            "cap_kind": kind, "cap_branch": "fallback",
            "x_dot_r_hat": "", "cap_half_angle_radians": "",
            "cap_support": "", "span_support": "",
            "current_ball_support": str(x_dot_r + n * epsilon),
            "current_ball_lb": "", "cap_lb_high_precision": "",
            "cap_would_prune": "0", "cap_gain": "",
            "cap_safety_slack": "", "gap_recovery": "",
            "support_contraction": "", 
            "certificate_failure": "1" if certificate_failure else "0",
            "support_oracle_mismatch": "0",
            "true_direction_support_violation": "0",
            "cap_lower_bound_violation": "0", "cap_below_current": "0",
            "cap_false_prune": "0",
            "cap_valid": "0",
        })
        return output

    support, branch, t = closed_form_support(kind, kappa, n, x_dot_r, s)
    span_support = span_candidate_support(kind, kappa, n, t)
    ball_support = x_dot_r + n * epsilon
    cap_lb = max(Decimal(0), n * n + ell * ell - 2 * ell * support)
    ball_lb = max(Decimal(0), n * n + ell * ell - 2 * ell * ball_support)
    cap_would_prune = cap_lb > threshold
    oracle_would_prune = bool_value(row, "oracle_would_prune")
    support_oracle_mismatch = abs(support - span_support) > oracle_tolerance
    true_direction_violation = true_support > support + comparison_tolerance
    lb_violation = cap_lb > exact + comparison_tolerance
    cap_below_current = cap_lb + comparison_tolerance < current_lb
    false_prune = cap_would_prune and not oracle_would_prune
    denominator = exact - current_lb
    gap_recovery = (
        (cap_lb - current_lb) / denominator
        if denominator > comparison_tolerance else None
    )
    cap_valid = not (
        certificate_failure or support_oracle_mismatch
        or true_direction_violation or lb_violation
    )
    cap_half_angle = (
        math.pi if kind in {"full_sphere", "s_zero"}
        else math.acos(max(-1.0, min(1.0, float(kappa))))
    )
    output.update({
        "kappa": "" if kappa is None else str(kappa),
        "cap_kind": kind, "cap_branch": branch,
        "x_dot_r_hat": "" if t is None else str(t),
        "cap_half_angle_radians": repr(cap_half_angle),
        "cap_support": str(support), "span_support": str(span_support),
        "current_ball_support": str(ball_support),
        "current_ball_lb": str(ball_lb),
        "cap_lb_high_precision": str(cap_lb),
        "cap_would_prune": "1" if cap_would_prune else "0",
        "cap_gain": str(cap_lb - current_lb),
        "cap_safety_slack": str(exact - cap_lb),
        "gap_recovery": "" if gap_recovery is None else str(gap_recovery),
        "support_contraction": str(ball_support - support),
        "certificate_failure": "1" if certificate_failure else "0",
        "support_oracle_mismatch": "1" if support_oracle_mismatch else "0",
        "true_direction_support_violation": "1" if true_direction_violation else "0",
        "cap_lower_bound_violation": "1" if lb_violation else "0",
        "cap_below_current": "1" if cap_below_current else "0",
        "cap_false_prune": "1" if false_prune else "0",
        "cap_valid": "1" if cap_valid else "0",
    })
    return output


def open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def quantile(values: Sequence[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = probability * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def write_geometry_summary(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    metrics = {
        "reconstruction_norm": "reconstruction_norm",
        "direction_error": "direction_error",
        "kappa": "kappa",
        "cap_half_angle_radians": "cap_half_angle_radians",
        "cap_gain": "cap_gain",
        "cap_safety_slack": "cap_safety_slack",
        "gap_recovery": "gap_recovery",
        "support_contraction": "support_contraction",
    }
    probabilities = [("p0", 0), ("p1", .01), ("p10", .1),
                     ("p50", .5), ("p90", .9), ("p99", .99), ("p100", 1)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "count"] + [p[0] for p in probabilities])
        writer.writeheader()
        for label, field in metrics.items():
            values = [float(row[field]) for row in rows if row.get(field, "") != ""]
            item = {"metric": label, "count": len(values)}
            item.update({name: quantile(values, probability) for name, probability in probabilities})
            writer.writerow(item)


def analyze(
    input_path: Path,
    output_dir: Path,
    comparison_tolerance: Decimal,
    oracle_tolerance: Decimal,
    certificate_tolerance: Decimal,
    min_extra_coverage: float,
    max_top_query_share: float,
    min_proper_fraction: float,
    min_boundary_fraction: float,
) -> Mapping[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open_csv(input_path) as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing required columns: " + ", ".join(sorted(missing)))
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            rows = [evaluate_row(row, comparison_tolerance, oracle_tolerance,
                                 certificate_tolerance) for row in reader]
    if not rows:
        raise ValueError("input contains no records")

    candidate_path = output_dir / "cap_candidate_table.csv.gz"
    with gzip.open(candidate_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts: MutableMapping[str, int] = defaultdict(int)
    per_query: MutableMapping[str, MutableMapping[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts["records"] += 1
        counts[f"kind_{row['cap_kind']}"] += 1
        counts[f"branch_{row['cap_branch']}"] += 1
        for field in ("current_would_prune", "raw_would_prune",
                      "oracle_would_prune", "cap_would_prune",
                      "certificate_failure", "support_oracle_mismatch",
                      "true_direction_support_violation", "cap_lower_bound_violation",
                      "cap_below_current",
                      "cap_false_prune", "cap_valid"):
            counts[field] += int(row[field])
        final_prune = int(row["current_would_prune"] == "1" or row["cap_would_prune"] == "1")
        extra = int(row["current_would_prune"] == "0" and row["cap_would_prune"] == "1")
        counts["final_would_prune"] += final_prune
        counts["cap_extra_prune"] += extra
        query = per_query[row["query_id"]]
        query["records"] += 1
        query["current_would_prune"] += int(row["current_would_prune"])
        query["raw_would_prune"] += int(row["raw_would_prune"])
        query["cap_would_prune"] += int(row["cap_would_prune"])
        query["oracle_would_prune"] += int(row["oracle_would_prune"])
        query["final_would_prune"] += final_prune
        query["cap_extra_prune"] += extra

    per_query_path = output_dir / "cap_coverage_by_query.csv"
    query_fields = ["query_id", "records", "current_would_prune", "raw_would_prune",
                    "cap_would_prune",
                    "final_would_prune", "oracle_would_prune", "cap_extra_prune",
                    "current_coverage", "raw_coverage", "cap_coverage",
                    "final_coverage", "oracle_coverage"]
    with per_query_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=query_fields)
        writer.writeheader()
        for query_id in sorted(per_query, key=int):
            item = dict(per_query[query_id])
            total = item["records"]
            item["query_id"] = query_id
            for name in ("current", "raw", "cap", "final", "oracle"):
                item[f"{name}_coverage"] = item[f"{name}_would_prune"] / total
            writer.writerow(item)

    write_geometry_summary(output_dir / "cap_geometry_summary.csv", rows)
    record_count = counts["records"]
    extra_coverage = counts["cap_extra_prune"] / record_count
    top_query_share = (
        max((item["cap_extra_prune"] for item in per_query.values()), default=0)
        / counts["cap_extra_prune"] if counts["cap_extra_prune"] else 0.0
    )
    proper_fraction = counts["kind_proper"] / record_count
    boundary_fraction = counts["branch_boundary"] / record_count
    query_count = len(per_query)
    zero_prune_fraction = {
        name: (
            sum(1 for item in per_query.values()
                if item[f"{name}_would_prune"] == 0) / query_count
            if query_count else None
        )
        for name in ("current", "raw", "cap", "final", "oracle")
    }
    safety_failures = sum(counts[name] for name in (
        "certificate_failure", "support_oracle_mismatch",
        "true_direction_support_violation", "cap_lower_bound_violation",
        "cap_below_current", "cap_false_prune"))
    reasons: List[str] = []
    if safety_failures:
        reasons.append("safety_or_certificate_gate_failed")
    if extra_coverage < min_extra_coverage:
        reasons.append("extra_coverage_below_threshold")
    if counts["cap_extra_prune"] and top_query_share > max_top_query_share:
        reasons.append("extra_pruning_too_concentrated")
    if proper_fraction < min_proper_fraction:
        reasons.append("proper_cap_fraction_below_threshold")
    if boundary_fraction < min_boundary_fraction:
        reasons.append("boundary_branch_fraction_below_threshold")
    decision = "go_candidate" if not reasons else "no_go"
    oracle_gap = counts["oracle_would_prune"] - counts["current_would_prune"]
    cap_recovery_ratio = (
        counts["cap_extra_prune"] / oracle_gap if oracle_gap > 0 else None
    )
    summary: Dict[str, object] = {
        "format": "v0_spherical_cap_phase1_summary",
        "format_version": 1,
        "decision": decision,
        "decision_reasons": reasons,
        "decimal_precision": DECIMAL_PRECISION,
        "comparison_tolerance": str(comparison_tolerance),
        "support_oracle_tolerance": str(oracle_tolerance),
        "certificate_tolerance": str(certificate_tolerance),
        "thresholds": {
            "min_extra_coverage": min_extra_coverage,
            "max_top_query_share": max_top_query_share,
            "min_proper_fraction": min_proper_fraction,
            "min_boundary_fraction": min_boundary_fraction,
        },
        "counts": dict(counts),
        "coverage": {
            "current": counts["current_would_prune"] / record_count,
            "raw": counts["raw_would_prune"] / record_count,
            "cap": counts["cap_would_prune"] / record_count,
            "final_current_or_cap": counts["final_would_prune"] / record_count,
            "oracle": counts["oracle_would_prune"] / record_count,
            "extra_cap": extra_coverage,
            "cap_to_oracle_recovery_ratio": cap_recovery_ratio,
            "top_query_extra_prune_share": top_query_share,
            "zero_prune_query_fraction": zero_prune_fraction,
        },
        "geometry": {
            "proper_fraction": proper_fraction,
            "boundary_fraction": boundary_fraction,
        },
        "cost_model": {
            "sampled_export_coordinate_passes": 2,
            "strict_phase2_incremental_cost_measured": False,
            "note": "go_candidate still requires a separate Phase-2 break-even audit",
        },
        "artifacts": {
            "candidate_table": candidate_path.name,
            "geometry_summary": "cap_geometry_summary.csv",
            "coverage_by_query": per_query_path.name,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "cap_data_quality.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "records": record_count,
            "diagnostic_invalid": record_count - counts["cap_valid"],
            "certificate_failure": counts["certificate_failure"],
            "support_oracle_mismatch": counts["support_oracle_mismatch"],
            "true_direction_support_violation": counts["true_direction_support_violation"],
            "cap_lower_bound_violation": counts["cap_lower_bound_violation"],
            "cap_below_current": counts["cap_below_current"],
            "cap_false_prune": counts["cap_false_prune"],
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--comparison-tolerance", default="1e-10", type=Decimal)
    parser.add_argument("--support-oracle-tolerance", default="1e-30", type=Decimal)
    parser.add_argument("--certificate-tolerance", default="1e-12", type=Decimal)
    parser.add_argument("--min-extra-coverage", default=0.001, type=float)
    parser.add_argument("--max-top-query-share", default=0.5, type=float)
    parser.add_argument("--min-proper-fraction", default=0.1, type=float)
    parser.add_argument("--min-boundary-fraction", default=0.1, type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(
        args.input, args.output_dir, args.comparison_tolerance,
        args.support_oracle_tolerance, args.certificate_tolerance,
        args.min_extra_coverage, args.max_top_query_share,
        args.min_proper_fraction, args.min_boundary_fraction)
    print(json.dumps({"decision": summary["decision"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
