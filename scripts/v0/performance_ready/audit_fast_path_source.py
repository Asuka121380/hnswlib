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
    print("raw_fast_v1 source audit OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
