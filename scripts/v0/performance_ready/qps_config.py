#!/usr/bin/env python3
"""Validate and resolve configurable V0 QPS experiment contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MODES = ("approx-no-retry", "approx-retry")
PREFETCHES = ("legacy", "gate")
RESOURCE_PROFILES = ("formal-exclusive", "exploratory-shared")
ROLES = ("formal", "exploratory")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MATRIX_KEYS = frozenset(("betas", "modes", "prefetches"))
ALLOWED_KEYS = frozenset((
    "schema_version", "experiment_name", "experiment_role", "dataset",
    "dimension", "query_start", "query_count", "k", "ef_search",
    "warmup_queries", "within_process_repeats", "blocks", "seed",
    "include_baseline", "betas", "modes", "prefetches", "cases",
))


class ConfigError(ValueError):
    """Raised when a QPS experiment contract is invalid."""


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def contract_sha256(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("contract_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _integer(
    requested: dict[str, Any], name: str, default: int | None = None,
    *, minimum: int = 0,
) -> int:
    value = requested.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _string(requested: dict[str, Any], name: str) -> str:
    value = requested.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _unique_strings(value: object, name: str, allowed: tuple[str, ...]) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            raise ConfigError(f"unknown {name} value: {item!r}")
        if item in result:
            raise ConfigError(f"duplicate {name} value: {item}")
        result.append(item)
    return result


def canonical_beta(value: object) -> str:
    if isinstance(value, bool):
        raise ConfigError("beta must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"beta must be numeric: {value!r}") from error
    if not math.isfinite(numeric) or numeric < 1.0:
        raise ConfigError("beta must be finite and >= 1.0")
    return format(numeric, ".15g")


def configuration_id(case: dict[str, str]) -> str:
    beta = case["beta"].replace(".", "p").replace("-", "m")
    return f"{case['mode']}-beta{beta}-{case['prefetch']}"


def _resolve_cases(requested: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    has_cases = "cases" in requested
    matrix_keys = MATRIX_KEYS.intersection(requested)
    if has_cases and matrix_keys:
        raise ConfigError("cases and betas/modes/prefetches are mutually exclusive")
    cases: list[dict[str, str]] = []
    if has_cases:
        raw_cases = requested["cases"]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ConfigError("cases must be a non-empty array")
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, dict):
                raise ConfigError(f"cases[{index}] must be an object")
            if set(raw_case) != {"beta", "mode", "prefetch"}:
                raise ConfigError(
                    f"cases[{index}] must contain only beta, mode, and prefetch")
            mode = raw_case["mode"]
            prefetch = raw_case["prefetch"]
            if mode not in MODES:
                raise ConfigError(f"unknown mode in cases[{index}]: {mode!r}")
            if prefetch not in PREFETCHES:
                raise ConfigError(
                    f"unknown prefetch in cases[{index}]: {prefetch!r}")
            cases.append({
                "beta": canonical_beta(raw_case["beta"]),
                "mode": mode,
                "prefetch": prefetch,
            })
        source = "explicit"
    else:
        if matrix_keys != MATRIX_KEYS:
            missing = sorted(MATRIX_KEYS - matrix_keys)
            raise ConfigError(
                "matrix configuration is missing: " + ", ".join(missing))
        raw_betas = requested["betas"]
        if not isinstance(raw_betas, list) or not raw_betas:
            raise ConfigError("betas must be a non-empty array")
        betas = [canonical_beta(value) for value in raw_betas]
        if len(set(betas)) != len(betas):
            raise ConfigError("betas contains duplicate canonical values")
        modes = _unique_strings(requested["modes"], "modes", MODES)
        prefetches = _unique_strings(
            requested["prefetches"], "prefetches", PREFETCHES)
        for beta in betas:
            for mode in modes:
                for prefetch in prefetches:
                    cases.append({
                        "beta": beta, "mode": mode, "prefetch": prefetch})
        source = "cartesian"
    identities = [
        (case["beta"], case["mode"], case["prefetch"]) for case in cases]
    if len(set(identities)) != len(identities):
        raise ConfigError("cases contains duplicate configurations")
    return source, cases


def resolve_config(
    requested: dict[str, Any], resource_profile: str | None = None,
) -> dict[str, Any]:
    if not isinstance(requested, dict):
        raise ConfigError("configuration root must be an object")
    unknown = sorted(set(requested) - ALLOWED_KEYS)
    if unknown:
        raise ConfigError("unknown configuration fields: " + ", ".join(unknown))
    if requested.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    name = _string(requested, "experiment_name")
    if not NAME_PATTERN.fullmatch(name):
        raise ConfigError(
            "experiment_name must use only letters, digits, '.', '_' and '-'")
    role = _string(requested, "experiment_role")
    if role not in ROLES:
        raise ConfigError(f"experiment_role must be one of: {', '.join(ROLES)}")
    dataset = _string(requested, "dataset")
    if resource_profile is not None:
        if resource_profile not in RESOURCE_PROFILES:
            raise ConfigError(
                f"resource_profile must be one of: {', '.join(RESOURCE_PROFILES)}")
        if role == "formal" and resource_profile != "formal-exclusive":
            raise ConfigError("formal experiments require formal-exclusive resources")

    include_baseline = requested.get("include_baseline", True)
    if not isinstance(include_baseline, bool):
        raise ConfigError("include_baseline must be boolean")
    if role == "formal" and not include_baseline:
        raise ConfigError("formal experiments require include_baseline=true")
    case_source, cases = _resolve_cases(requested)
    blocks = _integer(requested, "blocks", 5, minimum=1)
    repeats = _integer(
        requested, "within_process_repeats", 5, minimum=1)
    if role == "formal" and (blocks < 5 or repeats < 5):
        raise ConfigError(
            "formal experiments require blocks and within_process_repeats >= 5")

    k = _integer(requested, "k", 10, minimum=1)
    ef_search = _integer(requested, "ef_search", None, minimum=1)
    if ef_search < k:
        raise ConfigError("ef_search must be >= k")
    resolved: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": name,
        "experiment_role": role,
        "dataset": dataset,
        "dimension": _integer(requested, "dimension", 960, minimum=1),
        "query_start": _integer(requested, "query_start", 0, minimum=0),
        "query_count": _integer(requested, "query_count", 1000, minimum=1),
        "k": k,
        "ef_search": ef_search,
        "warmup_queries": _integer(
            requested, "warmup_queries", 100, minimum=0),
        "within_process_repeats": repeats,
        "blocks": blocks,
        "seed": _integer(requested, "seed", None, minimum=0),
        "include_baseline": include_baseline,
        "case_source": case_source,
        "cases": cases,
        "active_configuration_count": len(cases),
        "configuration_count": len(cases) + int(include_baseline),
        "expected_result_count": blocks * (len(cases) + int(include_baseline)),
    }
    if resource_profile is not None:
        resolved["resource_profile"] = resource_profile
        resolved["exclusive"] = resource_profile == "formal-exclusive"
    resolved["contract_sha256"] = contract_sha256(resolved)
    return resolved


def load_config(
    path: Path, resource_profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        requested = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read configuration {path}: {error}") from error
    return requested, resolve_config(requested, resource_profile)


def configurations(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if resolved["include_baseline"]:
        result.append({
            "id": "baseline", "method": "baseline",
            "beta": None, "prefetch": "none",
        })
    for case in resolved["cases"]:
        config = {
            "method": case["mode"],
            "beta": case["beta"],
            "prefetch": case["prefetch"],
        }
        config["id"] = configuration_id(case)
        result.append(config)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resource-profile", choices=RESOURCE_PROFILES)
    parser.add_argument("--requested-output", type=Path)
    parser.add_argument("--resolved-output", type=Path)
    parser.add_argument("--json", action="store_true",
                        help="print the complete resolved JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        requested, resolved = load_config(args.config, args.resource_profile)
    except ConfigError as error:
        raise SystemExit(f"invalid QPS configuration: {error}") from error
    if args.requested_output:
        _atomic_write_json(args.requested_output, requested)
    if args.resolved_output:
        _atomic_write_json(args.resolved_output, resolved)
    if args.json:
        print(json.dumps(resolved, indent=2, sort_keys=True))
    else:
        print(f"experiment={resolved['experiment_name']}")
        print(f"role={resolved['experiment_role']}")
        print(f"resource_profile={resolved.get('resource_profile', 'unbound')}")
        print(f"ef_search={resolved['ef_search']}")
        print(f"active_configurations={resolved['active_configuration_count']}")
        print(f"blocks={resolved['blocks']}")
        print(f"expected_result_count={resolved['expected_result_count']}")
        print(f"contract_sha256={resolved['contract_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
