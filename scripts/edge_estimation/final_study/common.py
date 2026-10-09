from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from scripts.edge_estimation.contracts import sha256_file, load_strict_json

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def load(path):
    return load_strict_json(Path(path))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def seal(path, value):
    if "content_sha256" in value:
        raise ValueError("cannot reseal a payload containing content_sha256")
    payload = {**value, "content_sha256": digest(value)}
    write(path, payload)
    return payload

def unseal(path):
    value = load(path)
    checksum = value.pop("content_sha256", None)
    if checksum is None or digest(value) != checksum:
        raise ValueError(f"sealed content mismatch: {path}")
    return value

def identity(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}

def verify_file(entry):
    actual = identity(entry["path"])
    if actual["sha256"] != entry["sha256"] or actual["bytes"] != entry["bytes"]:
        raise ValueError(f"file identity mismatch: {entry['path']}")
    return Path(entry["path"])

def seed(*values):
    return int(digest(list(values))[:16], 16)

def name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("invalid dataset/study identifier")
    return value

def positive(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value

def cli(main):
    try:
        main()
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        raise SystemExit(f"FINAL_STUDY_FAILED: {error}") from error
