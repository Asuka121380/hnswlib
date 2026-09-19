#!/usr/bin/env python3
"""Static guard against strict-only operations leaking into raw_fast_v1."""

from __future__ import annotations

import argparse
from pathlib import Path


def slice_between(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", required=True, type=Path)
    parser.add_argument("--residual-header", type=Path)
    parser.add_argument("--search-header", type=Path)
    args = parser.parse_args()
    text = args.header.read_text(encoding="utf-8")
    fast_lut = slice_between(text, "class V0ApproxQueryLut", "enum class V0BoundStatus")
    fast_estimator = slice_between(
        text, "class EdgeQuantV0ApproxQueryContext",
        "class EdgeQuantV0QueryContext")
    forbidden = (
        "std::sqrt", "std::nextafter", "directionError(",
        "numericPadding(", "floatSquaredL2PaddingUpper",
        "V0BoundResult", "codebookCentroid(", "edge.code(",
    )
    failures = [token for token in forbidden
                if token in fast_lut or token in fast_estimator]
    if failures:
        print("raw_fast_v1 source audit FAILED")
        for token in failures:
            print(f"  forbidden token: {token}")
        return 1
    required = (
        "native_codebook", "codeDataUnchecked",
        "edgeLengthNativeUnchecked", "anchorProjectionNativeUnchecked",
    )
    missing = [token for token in required
               if token not in fast_lut and token not in fast_estimator]
    if missing:
        print("raw_fast_v1 source audit FAILED")
        for token in missing:
            print(f"  missing fast-path token: {token}")
        return 1
    if args.residual_header:
        residual = args.residual_header.read_text(encoding="utf-8")
        query = slice_between(residual, "class V0ResidualQueryContext", "// Offline reference")
        banned = ("std::chrono", "std::ofstream", "std::nextafter", "std::sqrt")
        failures = [token for token in banned if token in query]
        if failures:
            print("residual query-path source audit FAILED: " + ", ".join(failures))
            return 1
    if args.search_header:
        source = args.search_header.read_text(encoding="utf-8")
        path = slice_between(source, "if (v0_residual_config != nullptr &&",
                             "#endif\n#ifdef HNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING")
        banned = ("std::chrono", "std::ofstream", "unordered_map", "std::nextafter")
        failures = [token for token in banned if token in path]
        if failures:
            print("residual search-path source audit FAILED: " + ", ".join(failures))
            return 1
    print("raw_fast_v1 source audit OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
