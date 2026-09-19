"""Resolve all real assets, commit state and query split before experiments."""

import argparse
import subprocess
from pathlib import Path

from common import git_state, mark_complete, query_split, read_json, resolved_asset, sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    assets = read_json(args.assets)
    if config.get("schema_version") != 1 or config.get("dataset") != "gist1m":
        raise ValueError("unsupported residual prototype configuration")
    repo = Path(__file__).resolve().parents[3]
    state = git_state(repo)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["base_commit"], state["head"]],
        cwd=repo, check=False
    ).returncode == 0
    if not ancestor or state["dirty"]:
        raise ValueError("experiment requires a clean implementation descended from base_commit")
    required = ("index", "sidecar", "queries", "trace")
    resolved = {}
    for name in required:
        path = resolved_asset(assets, name)
        resolved[name] = {"path": str(path), "sha256": sha256(path)}
    for name in ("ground_truth", "base", "mapping", "dataset_config"):
        if name in assets:
            path = resolved_asset(assets, name)
            resolved[name] = {"path": str(path), "sha256": sha256(path)}
    split = query_split(1000, config["split_counts"], config["split_seed"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "split.json", split)
    manifest = {"schema_version": 1, "config": config,
                "config_sha256": sha256(args.config), "assets": resolved,
                "git": state, "split_path": str((output / "split.json").resolve()),
                "split_sha256": sha256(output / "split.json")}
    write_json(output / "manifest.json", manifest)
    mark_complete(output, {"config": manifest["config_sha256"]},
                  {"manifest": sha256(output / "manifest.json")})


if __name__ == "__main__":
    main()
