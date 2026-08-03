#!/usr/bin/env python3
"""Validate and analyse one v0_search_runner shadow-mode run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_SHADOW_COLUMNS = {
    "query_id",
    "current_node_id",
    "candidate_id",
    "bound_status",
    "current_squared_distance",
    "threshold",
    "approximate_squared_distance",
    "error_radius",
    "lower_bound",
    "shadow_exact_squared_distance",
    "would_prune",
    "lower_bound_valid",
    "lower_bound_violation",
    "false_prune",
}

ALPHAS = np.round(np.arange(0.0, 1.0001, 0.1), 1)
QUANTILES = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-query-count", type=int)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--sampling-relative-tolerance", type=float, default=0.10)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(np.int64).ne(0)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def add_derived(chunk: pd.DataFrame) -> pd.DataFrame:
    for column in (
        "would_prune",
        "lower_bound_valid",
        "lower_bound_violation",
        "false_prune",
    ):
        chunk[column] = as_bool(chunk[column])
    chunk["margin"] = chunk["shadow_exact_squared_distance"] - chunk["threshold"]
    chunk["raw_margin"] = chunk["approximate_squared_distance"] - chunk["threshold"]
    chunk["raw_gap"] = (
        chunk["shadow_exact_squared_distance"]
        - chunk["approximate_squared_distance"]
    )
    chunk["safe_gap"] = chunk["shadow_exact_squared_distance"] - chunk["lower_bound"]
    chunk["oracle_prunable"] = chunk["shadow_exact_squared_distance"] > chunk["threshold"]
    chunk["raw_prunable"] = chunk["approximate_squared_distance"] > chunk["threshold"]
    chunk["safe_prunable"] = chunk["lower_bound"] > chunk["threshold"]
    denominator = np.maximum(np.abs(chunk["margin"].to_numpy(float)), 1e-12)
    chunk["radius_margin_ratio"] = chunk["error_radius"].to_numpy(float) / denominator
    finite = np.isfinite(
        chunk[
            [
                "threshold",
                "approximate_squared_distance",
                "error_radius",
                "lower_bound",
                "shadow_exact_squared_distance",
            ]
        ].to_numpy(float)
    ).all(axis=1)
    chunk["analysis_valid"] = chunk["lower_bound_valid"] & finite
    return chunk


def quantile_dict(values: pd.Series) -> dict[str, float | None]:
    finite = values[np.isfinite(values.to_numpy(float))]
    if finite.empty:
        return {f"p{int(q * 100):02d}": None for q in QUANTILES}
    result = finite.quantile(QUANTILES)
    return {f"p{int(q * 100):02d}": float(result.loc[q]) for q in QUANTILES}


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    values.sort()
    if values.size == 0:
        return values, values
    return values, np.arange(1, values.size + 1) / values.size


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def classify(oracle: float, raw: float, safe: float) -> tuple[str, str]:
    opportunity = oracle
    estimator_loss = max(oracle - raw, 0.0)
    safety_loss = max(raw - safe, 0.0)
    if opportunity <= 0.01:
        return (
            "搜索机会不足",
            "先检查 bound 的触发位置、阈值形成过程或仅在更有利的搜索阶段启用。",
        )
    if estimator_loss >= safety_loss and raw < 0.5 * oracle:
        return (
            "近似距离区分度不足",
            "优先改进距离估计器或量化表示，再重复 observe-only shadow。",
        )
    if safety_loss > estimator_loss and safe < 0.5 * raw:
        return (
            "安全半径过保守",
            "下一轮只增加 radius 分解字段，定位 quantisation/stored/numeric padding 的贡献。",
        )
    return (
        "混合瓶颈或收益集中",
        "按 query/bucket 检查收益集中度，并评估低成本 selective gate。",
    )


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    summary = load_json(run_dir / "summary.json")
    metadata = load_json(run_dir / "metadata.json")
    query = pd.read_csv(run_dir / "query_metrics.csv")
    shadow_path = run_dir / "shadow_records.csv"
    if not shadow_path.exists():
        shadow_path = run_dir / "shadow_records.csv.gz"
    if not shadow_path.exists():
        raise FileNotFoundError("shadow_records.csv[.gz] is missing")

    failures: list[str] = []
    warnings: list[str] = []
    expected_queries = args.expected_query_count
    if expected_queries is not None and int(summary["query_count"]) != expected_queries:
        failures.append(
            f"query_count={summary['query_count']} (expected {expected_queries})"
        )
    if summary.get("status") != "valid":
        failures.append(f"summary.status={summary.get('status')!r}")
    if metadata.get("mode") != "shadow":
        failures.append(f"metadata.mode={metadata.get('mode')!r}, expected 'shadow'")
    if metadata.get("real_pruning_compiled") is not False:
        failures.append("real_pruning_compiled must be false")
    if metadata.get("shadow_validation_compiled") is not True:
        failures.append("shadow_validation_compiled must be true")
    for key in ("mismatch_queries", "exact_distance_saved", "lower_bound_violation", "false_prune"):
        if int(summary.get(key, -1)) != 0:
            failures.append(f"summary.{key}={summary.get(key)} (expected 0)")
    if int(summary.get("bound_evaluated", 0)) <= 0:
        failures.append("bound_evaluated must be positive")
    attempted_bounds = (
        int(summary.get("bound_evaluated", 0))
        + int(summary.get("exact_fallback", 0))
        + int(summary.get("exact_only_fallback", 0))
    )
    if int(summary.get("shadow_records_seen", -1)) != attempted_bounds:
        failures.append(
            "shadow_records_seen != bound_evaluated + exact_fallback + "
            "exact_only_fallback"
        )
    if len(query) != int(summary.get("query_count", -1)):
        failures.append("query_metrics row count != summary.query_count")
    if "results_equal" in query and not as_bool(query["results_equal"]).all():
        failures.append("at least one query has results_equal=false")
    if not np.allclose(
        query["baseline_recall_at_k"], query["v0_recall_at_k"], rtol=0.0, atol=1e-15
    ):
        failures.append("baseline and V0 recall differ")
    if int(query["bound_evaluated"].sum()) != int(summary.get("bound_evaluated", -1)):
        failures.append("sum(query bound_evaluated) != summary.bound_evaluated")

    parquet_path = output_dir / "candidate_metrics.parquet"
    writer: pq.ParquetWriter | None = None
    sampled_parts: list[pd.DataFrame] = []
    closure_mismatch = 0
    would_prune_mismatch = 0
    sampled_rows = 0
    valid_rows = 0
    min_safe_gap = math.inf

    for chunk in pd.read_csv(shadow_path, chunksize=args.chunksize):
        missing = REQUIRED_SHADOW_COLUMNS.difference(chunk.columns)
        if missing:
            raise ValueError(f"shadow CSV missing columns: {sorted(missing)}")
        chunk = add_derived(chunk)
        expected_lower = np.maximum(
            0.0,
            chunk["approximate_squared_distance"].to_numpy(float)
            - chunk["error_radius"].to_numpy(float),
        )
        closure_mismatch += int(
            (~np.isclose(chunk["lower_bound"], expected_lower, rtol=1e-10, atol=1e-9)).sum()
        )
        would_prune_mismatch += int(
            (chunk["would_prune"] != chunk["safe_prunable"]).sum()
        )
        sampled_rows += len(chunk)
        valid = chunk.loc[chunk["analysis_valid"]].copy()
        valid_rows += len(valid)
        if not valid.empty:
            min_safe_gap = min(min_safe_gap, float(valid["safe_gap"].min()))
            sampled_parts.append(valid)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()
    if sampled_rows == 0:
        failures.append("shadow CSV contains no records")
    if closure_mismatch:
        failures.append(f"lower-bound closure mismatches={closure_mismatch}")
    if would_prune_mismatch:
        failures.append(f"would_prune mismatches={would_prune_mismatch}")
    if valid_rows == 0:
        failures.append("no analysis-valid shadow records")

    written = int(summary.get("shadow_records_written", -1))
    if sampled_rows != written:
        failures.append(f"CSV rows={sampled_rows} != shadow_records_written={written}")
    modulus = int(metadata.get("shadow_sample_modulus", 1))
    seen = int(summary.get("shadow_records_seen", 0))
    expected_written = seen / modulus if modulus else 0.0
    if expected_written > 0:
        relative_error = abs(written - expected_written) / expected_written
        if relative_error > args.sampling_relative_tolerance:
            failures.append(
                f"sampling ratio differs from 1/{modulus}: relative error={relative_error:.3%}"
            )

    data = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    if data.empty:
        raise RuntimeError("no valid candidate data; see data-quality failures")

    coverage = {
        "oracle": float(data["oracle_prunable"].mean()),
        "raw": float(data["raw_prunable"].mean()),
        "safe": float(data["safe_prunable"].mean()),
    }
    coverage["estimator_loss"] = coverage["oracle"] - coverage["raw"]
    coverage["safety_loss"] = coverage["raw"] - coverage["safe"]
    full_query_zero_prune_fraction = float((query["bound_pruned"] == 0).mean())
    full_query_safe_coverage = np.divide(
        query["bound_pruned"].to_numpy(float),
        query["bound_evaluated"].to_numpy(float),
        out=np.zeros(len(query), dtype=float),
        where=query["bound_evaluated"].to_numpy(float) > 0,
    )

    distributions = {
        name: quantile_dict(data[name])
        for name in ("margin", "raw_gap", "safe_gap", "error_radius", "radius_margin_ratio")
    }
    alpha_rows: list[dict[str, float]] = []
    approx = data["approximate_squared_distance"].to_numpy(float)
    radius = data["error_radius"].to_numpy(float)
    threshold = data["threshold"].to_numpy(float)
    exact = data["shadow_exact_squared_distance"].to_numpy(float)
    for alpha in ALPHAS:
        lower = np.maximum(0.0, approx - alpha * radius)
        alpha_rows.append(
            {
                "alpha": float(alpha),
                "coverage": float(np.mean(lower > threshold)),
                "empirical_violation": float(np.mean(lower > exact + 1e-9)),
                "empirical_false_prune": float(np.mean((lower > threshold) & (exact <= threshold))),
            }
        )
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(output_dir / "alpha_sensitivity.csv", index=False)

    diagnostic, recommendation = classify(coverage["oracle"], coverage["raw"], coverage["safe"])

    summary_row = {
        "run_id": metadata.get("run_id"),
        "query_count": summary.get("query_count"),
        "ef_search": metadata.get("ef_search"),
        "bound_evaluated": summary.get("bound_evaluated"),
        "sampled_rows": sampled_rows,
        "analysis_valid_rows": valid_rows,
        "oracle_coverage": coverage["oracle"],
        "raw_coverage": coverage["raw"],
        "safe_coverage": coverage["safe"],
        "estimator_loss": coverage["estimator_loss"],
        "safety_loss": coverage["safety_loss"],
        "full_query_zero_prune_fraction": full_query_zero_prune_fraction,
        "full_query_safe_coverage_median": float(np.median(full_query_safe_coverage)),
        "min_safe_gap": min_safe_gap,
        "diagnosis": diagnostic,
        "gate_passed": not failures,
    }
    pd.DataFrame([summary_row]).to_csv(output_dir / "diagnostic_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels = ["Oracle: D>T", "Raw: D_hat>T", "Safe: L>T"]
    values = [coverage["oracle"], coverage["raw"], coverage["safe"]]
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.bar_label(bars, labels=[f"{value:.4%}" for value in values], padding=3)
    ax.set_ylabel("Coverage among valid sampled candidates")
    ax.set_ylim(0, max(values + [0.01]) * 1.18)
    ax.set_title("Pruning opportunity waterfall")
    save_figure(fig, figures_dir / "01_coverage_waterfall")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x, y = ecdf(data["margin"].to_numpy(float))
    ax.plot(x, y, color="#4C78A8")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("margin = D - T")
    ax.set_ylabel("ECDF")
    ax.set_title("Exact pruning opportunity distribution")
    save_figure(fig, figures_dir / "02_margin_ecdf")

    x = data["raw_margin"].to_numpy(float)
    y = data["error_radius"].to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        xlo, xhi = np.quantile(x[finite], [0.005, 0.995])
        ylo, yhi = np.quantile(y[finite], [0.005, 0.995])
        keep = finite & (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    else:
        keep = finite
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    if keep.any():
        hb = ax.hexbin(x[keep], y[keep], gridsize=55, bins="log", mincnt=1, cmap="viridis")
        fig.colorbar(hb, ax=ax, label="log10(count)")
        line_max = max(0.0, min(float(np.max(x[keep])), float(np.max(y[keep]))))
        ax.plot([0.0, line_max], [0.0, line_max], "r--", linewidth=1.2, label="D_hat-T = r")
        ax.legend()
    ax.set_xlabel("Raw margin = D_hat - T")
    ax.set_ylabel("Error radius r")
    ax.set_title("Raw margin versus conservative radius")
    save_figure(fig, figures_dir / "03_raw_margin_vs_radius")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for column, color in (("raw_gap", "#F58518"), ("safe_gap", "#54A24B")):
        xx, yy = ecdf(data[column].to_numpy(float))
        ax.plot(xx, yy, label=column, color=color)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Distance gap")
    ax.set_ylabel("ECDF")
    ax.legend()
    ax.set_title("Approximation gap versus safety gap")
    save_figure(fig, figures_dir / "04_raw_safe_gap_ecdf")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.5), sharex=True)
    ax1.plot(alpha_df["alpha"], alpha_df["coverage"], marker="o", color="#4C78A8")
    ax1.set_ylabel("Coverage")
    ax1.set_title("Offline alpha sensitivity (diagnostic only)")
    ax2.plot(alpha_df["alpha"], alpha_df["empirical_violation"], marker="o", label="violation")
    ax2.plot(alpha_df["alpha"], alpha_df["empirical_false_prune"], marker="s", label="false prune")
    ax2.set_xlabel("alpha in max(0, D_hat - alpha*r)")
    ax2.set_ylabel("Empirical risk")
    ax2.legend()
    save_figure(fig, figures_dir / "05_alpha_coverage_risk")

    quality = {
        "gate_passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "closure_mismatch": closure_mismatch,
        "would_prune_mismatch": would_prune_mismatch,
        "sampled_rows": sampled_rows,
        "analysis_valid_rows": valid_rows,
        "sampling_modulus": modulus,
        "attempted_bounds_accounted": attempted_bounds,
        "summary": summary,
        "metadata": metadata,
        "coverage": coverage,
        "distributions": distributions,
    }
    with (output_dir / "data_quality.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    report = f"""# V0 shadow 最小诊断结果

## 数据门禁

- 结论：**{'通过' if not failures else '未通过'}**
- run ID：`{metadata.get('run_id')}`
- queries：{summary.get('query_count')}
- efSearch：{metadata.get('ef_search')}
- 全量 bound attempts：{int(summary.get('bound_evaluated', 0)):,}
- 候选明细采样：{sampled_rows:,} 行（1/{modulus} 确定性采样）
- analysis-valid：{valid_rows:,} 行
- lower-bound violation：{summary.get('lower_bound_violation')}
- false prune：{summary.get('false_prune')}

失败项：{'; '.join(failures) if failures else '无'}  
警告项：{'; '.join(warnings) if warnings else '无'}

## Coverage 漏斗

| 指标 | 比例 |
|---|---:|
| Oracle coverage `P(D>T)` | {coverage['oracle']:.6%} |
| Raw coverage `P(D_hat>T)` | {coverage['raw']:.6%} |
| Safe coverage `P(L>T)` | {coverage['safe']:.6%} |
| Estimator loss | {coverage['estimator_loss']:.6%} |
| Safety loss | {coverage['safety_loss']:.6%} |

全量 query metrics 中零 would-prune 查询比例：{full_query_zero_prune_fraction:.2%}。这项使用全量 query 汇总，不受候选明细采样影响。

## 初步判断

**{diagnostic}。** {recommendation}

该判断是候选级诊断，不等价于真实剪枝后的 Recall 或性能结论。`alpha<1` 的结果只能用于提出假设，不能直接用于上线剪枝。

## 输出

- `diagnostic_summary.csv`：核心指标；
- `candidate_metrics.parquet`：带派生字段的候选明细；
- `alpha_sensitivity.csv`：alpha coverage/risk；
- `data_quality.json`：门禁、元数据和分位数；
- `figures/`：5 类 PNG/PDF 图。
"""
    (output_dir / "analysis.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary_row, ensure_ascii=False, indent=2))
    if failures:
        print("Gate failed: " + "; ".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
