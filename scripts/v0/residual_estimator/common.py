"""Shared, fail-closed contracts for the residual prototype."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=target.name + ".",
        suffix=".partial", delete=False
    ) as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        temporary = Path(output.name)
    os.replace(temporary, target)


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolved_asset(assets: dict, name: str) -> Path:
    raw = assets.get(name)
    if isinstance(raw, str):
        path, expected = raw, None
    elif isinstance(raw, dict):
        path, expected = raw.get("path"), raw.get("sha256")
    else:
        raise ValueError(f"missing asset {name}")
    if not path:
        raise ValueError(f"missing path for asset {name}")
    target = Path(path).expanduser().resolve(strict=True)
    if not target.is_file():
        raise ValueError(f"asset is not a file: {target}")
    actual = sha256(target)
    if expected and actual.lower() != expected.lower():
        raise ValueError(f"SHA mismatch for asset {name}: {target}")
    return target


def git_state(repo: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True
        ).strip()
    return {"head": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain"))}


def query_split(query_count: int, counts: list[int], seed: int) -> dict[str, list[int]]:
    if len(counts) != 3 or sum(counts) != query_count or min(counts) < 0:
        raise ValueError("split counts must cover each query exactly once")
    ids = list(range(query_count))
    random.Random(seed).shuffle(ids)
    a, b, _ = counts
    return {"development": sorted(ids[:a]),
            "selection": sorted(ids[a:a + b]),
            "audit": sorted(ids[a + b:])}


def mark_complete(output: Path, inputs: dict, outputs: dict) -> None:
    write_json(output / "complete.json", {
        "schema_version": 1, "inputs": inputs, "outputs": outputs
    })
