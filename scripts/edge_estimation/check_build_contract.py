from __future__ import annotations

import argparse
import json
import subprocess
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path, binaries: list[Path] | None = None) -> dict[str, object]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    uq_entries = [entry for entry in entries if "uq_" in entry.get("file", "")]
    commands = [entry["command"] for entry in uq_entries]
    problems: list[str] = []
    if not uq_entries:
        problems.append("no unified edge-estimation translation units found")
    for command in commands:
        if "-std=c++17" not in command and "/std:c++17" not in command:
            problems.append("unified target is not compiled as C++17")
            break
    source_root = path.resolve().parent.parent
    git: dict[str, object] = {"available": False}
    try:
        def git_value(*arguments: str) -> str:
            return subprocess.run(
                ["git", "-C", str(source_root), *arguments], check=True,
                text=True, capture_output=True).stdout.strip()
        dirty = bool(git_value("status", "--porcelain"))
        git = {"available": True, "head": git_value("rev-parse", "HEAD"),
               "tree": git_value("rev-parse", "HEAD^{tree}"), "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        problems.append("source git identity unavailable")
    compilers = sorted({entry["command"].split()[0] for entry in uq_entries})
    compiler_versions: dict[str, str] = {}
    for compiler in compilers:
        try:
            compiler_versions[compiler] = subprocess.run(
                [compiler, "--version"], check=True, text=True,
                capture_output=True).stdout.splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            compiler_versions[compiler] = "unavailable"
            problems.append(f"compiler version unavailable: {compiler}")
    binary_entries = []
    for binary in binaries or []:
        resolved = binary.resolve()
        if not resolved.is_file():
            problems.append(f"binary missing: {resolved}")
            continue
        binary_entries.append({"path": str(resolved), "size": resolved.stat().st_size,
                               "sha256": sha256(resolved)})
    return {
        "schema_version": 1,
        "valid": not problems,
        "translation_unit_count": len(uq_entries),
        "problems": problems,
        "commands": commands,
        "compiler_executables": compilers,
        "compiler_versions": compiler_versions,
        "binaries": binary_entries,
        "source": git,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("compile_commands")
    parser.add_argument("--out")
    parser.add_argument("--binary", action="append", default=[])
    args = parser.parse_args()
    report = inspect(Path(args.compile_commands), [Path(value) for value in args.binary])
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
