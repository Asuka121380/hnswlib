#!/usr/bin/env python3
"""Synthetic property tests for EQ-RCP Stage 3 progressive quantization."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "eq_rcp"
sys.path.insert(0, str(SCRIPT_DIR))

from stage2_1_quantization import UnitDirectionQuantizer  # noqa: E402
from stage3_quantization import (  # noqa: E402
    ProgressiveGainShapeQuantizer,
    _balanced_blocks,
    evaluate_stopping,
    fit_stopping_radius,
    projection_metrics,
    projection_payload,
    save_progressive_model,
)
from fit_operational_rate_distortion import summarize as summarize_rate_distortion  # noqa: E402
from run_stage3_progressive import _require_ready_input  # noqa: E402


def run_tests() -> None:
    blocks_28 = _balanced_blocks(960, 28, np.linspace(1.0, 2.0, 960), "joint_objective")
    assert len(blocks_28) == 28
    assert sorted(np.concatenate(blocks_28).tolist()) == list(range(960))
    assert max(map(len, blocks_28)) - min(map(len, blocks_28)) <= 1

    rng = np.random.default_rng(20260811)
    edges = rng.normal(size=(320, 12)).astype(np.float32)
    edges[0] = 0.0
    weights = rng.uniform(0.2, 2.0, size=len(edges))
    x = rng.normal(size=(500, 12)).astype(np.float32)
    inverse = rng.integers(0, len(edges), size=len(x))
    current = rng.uniform(1.0, 3.0, size=len(x))
    edge_length = np.sum(edges[inverse].astype(np.float64) ** 2, axis=1)
    exact_y = np.einsum("ij,ij->i", x.astype(np.float64), edges[inverse].astype(np.float64))
    exact_distance = current + edge_length - 2.0 * exact_y
    threshold = np.quantile(exact_distance, 0.55) + 0.05 * rng.normal(size=len(x))
    good = exact_distance <= threshold
    prunable = ~good
    spec = {
        "subquantizers": 4,
        "centroids": 8,
        "max_iter": 5,
        "batch_size": 64,
        "opq_iterations": 1,
    }
    frozen = UnitDirectionQuantizer.fit(
        "gain_shape_opq", edges, weights, spec, 17, "opq"
    )
    with tempfile.TemporaryDirectory(prefix="eq-rcp-stage3-") as temp:
        root = Path(temp)
        frozen_path = root / "frozen.npz"
        np.savez_compressed(frozen_path, **frozen.artifact_arrays())
        flat = ProgressiveGainShapeQuantizer.from_frozen_flat(frozen_path)
        flat_result = flat.reconstruct(edges)
        frozen_result, frozen_codes = frozen.reconstruct(edges)
        assert np.array_equal(flat_result["coarse_codes"], frozen_codes)
        assert np.max(np.abs(flat_result["full_reconstruction"] - frozen_result)) < 2e-5
        assert flat.coarse_bytes == 4 and flat.residual_bytes == 0

        full_improvements = []
        for index, strategy in enumerate(
            ["uniform", "operational_rate_distortion", "joint_objective", "greedy_residual"]
        ):
            model = ProgressiveGainShapeQuantizer.fit(
                edges,
                weights,
                frozen.base.rotation,
                coarse_bytes=3,
                residual_bytes=1,
                allocation_strategy=strategy,
                operational_x=x,
                seed=43 + index,
                spec={"centroids": 8, "max_iter": 5, "batch_size": 64},
                layout_name="progressive_3_1",
            )
            reconstructed = model.reconstruct(edges)
            assert reconstructed["coarse_codes"].shape == (len(edges), 3)
            assert reconstructed["residual_codes"].shape == (len(edges), 1)
            assert model.coarse_bytes + model.residual_bytes == 4
            original_norm = np.linalg.norm(edges.astype(np.float64), axis=1)
            coarse_norm = np.linalg.norm(reconstructed["coarse_reconstruction"], axis=1)
            full_norm = np.linalg.norm(reconstructed["full_reconstruction"], axis=1)
            assert np.max(np.abs(original_norm - coarse_norm)) < 2e-5
            assert np.max(np.abs(original_norm - full_norm)) < 2e-5
            assert np.array_equal(reconstructed["full_reconstruction"][0], np.zeros(12))

            payload = projection_payload(
                model, edges, inverse, x, current, edge_length, threshold, good, prunable
            )
            metrics = projection_metrics(payload)
            full_improvements.append(
                metrics["coarse_projection_abs_q99"] - metrics["full_projection_abs_q99"]
            )
            radius = fit_stopping_radius(payload["coarse_projection_error"], 0.99)
            stopping = evaluate_stopping(payload, radius)
            assert 0.0 <= stopping["coarse_only_fraction"] <= 1.0
            artifact_path = root / f"{strategy}.npz"
            metadata = save_progressive_model(model, artifact_path, {"seed": 43 + index})
            assert artifact_path.is_file()
            assert metadata["total_code_bytes"] == 4
            assert metadata["residual_contract"] == "gain_shape_tangent_direction_residual"
            reloaded = ProgressiveGainShapeQuantizer.load(artifact_path, metadata)
            reloaded_result = reloaded.reconstruct(edges)
            assert np.array_equal(
                reconstructed["coarse_codes"], reloaded_result["coarse_codes"]
            )
            assert np.array_equal(
                reconstructed["residual_codes"], reloaded_result["residual_codes"]
            )
            assert np.max(
                np.abs(
                    reconstructed["full_reconstruction"]
                    - reloaded_result["full_reconstruction"]
                )
            ) < 1e-7
        assert np.mean(full_improvements) > 0.0

        rate = summarize_rate_distortion(
            {
                "summary": {
                    "flat_32__frozen_flat": {
                        "mean_coarse_projection_abs_q99": 1.0,
                        "mean_full_projection_abs_q99": 1.0,
                    },
                    "progressive_3_1__uniform": {
                        "mean_coarse_projection_abs_q99": 1.2,
                        "mean_full_projection_abs_q99": 1.02,
                    },
                }
            }
        )
        assert rate["rows"][1]["coarse_q99_reduction_per_residual_byte"] > 0.0

        try:
            _require_ready_input(
                {
                    "format": "eq_rcp_stage3_input_manifest",
                    "version": 1,
                    "status": "PENDING_FRESH_STAGE3_EVALUATION_QUERIES",
                    "experiment_ready": False,
                }
            )
        except ValueError as error:
            assert "bind fresh evaluation queries" in str(error)
        else:
            raise AssertionError("pending Stage 3 input did not fail closed")

        try:
            ProgressiveGainShapeQuantizer.fit(
                edges,
                weights,
                frozen.base.rotation,
                3,
                1,
                "unknown",
                x,
                1,
                {"centroids": 8, "max_iter": 2, "batch_size": 32},
                "invalid",
            )
        except ValueError as error:
            assert "unsupported Stage 3 allocation" in str(error)
        else:
            raise AssertionError("unknown Stage 3 allocation did not fail closed")
    print("eq_rcp_progressive_test: PASS")


if __name__ == "__main__":
    run_tests()
