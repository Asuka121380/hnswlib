#!/usr/bin/env python3
"""Reject a CMake build tree that is unsafe for V0 validation or timing."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True,
                        choices=("reference", "performance", "residual-reference",
                                 "residual-quality", "residual-performance"))
    parser.add_argument(
        "--native-arch", choices=("off", "on"), default="off",
        help=("Expected HNSWLIB_ENABLE_NATIVE_ARCH value. Performance "
              "builds default to portable/off for backward compatibility."),
    )
    args = parser.parse_args()
    cache_path = args.build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        raise SystemExit(f"missing CMake cache: {cache_path}")
    cache = read_cache(cache_path)

    required = {
        "CMAKE_BUILD_TYPE": "Release",
        "HNSWLIB_ENABLE_EDGE_QUANT_V0": "ON",
        "HNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING": "ON",
    }
    residual = args.contract.startswith("residual-")
    reference = args.contract in ("reference", "residual-reference", "residual-quality")
    if residual:
        required["HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR"] = "ON"
        required["HNSWLIB_ENABLE_V0_RESIDUAL_REAL_PRUNING"] = (
            "OFF" if args.contract == "residual-reference" else "ON")
    if reference:
        if args.native_arch != "off":
            parser.error("reference builds must use --native-arch off")
        required.update({
            "HNSWLIB_ENABLE_V0_STRICT_FP_CONTRACT": "ON",
            "HNSWLIB_ENABLE_NATIVE_ARCH": "OFF",
        })
    else:
        required.update({
            "HNSWLIB_ENABLE_V0_STRICT_FP_CONTRACT": "OFF",
            "HNSWLIB_PERFORMANCE_COMPARABLE_FLAGS": "ON",
            "HNSWLIB_ENABLE_NATIVE_ARCH": args.native_arch.upper(),
            "HNSWLIB_ENABLE_BASELINE_TRACE": "OFF",
            "HNSWLIB_ENABLE_V0_SHADOW_VALIDATION": "OFF",
            "HNSWLIB_ENABLE_V0_APPROX_SHADOW": "OFF",
        })

    failures = []
    for key, expected in required.items():
        actual = cache.get(key, "<unset>")
        if actual.upper() != expected.upper():
            failures.append(f"{key}: expected {expected}, got {actual}")
    if failures:
        print(f"{args.contract} build contract FAILED")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"{args.contract} build contract OK: {args.build_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
