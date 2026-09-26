from __future__ import annotations

import importlib.util
import platform
from typing import Any


def backend_capabilities() -> list[dict[str, Any]]:
    faiss = importlib.util.find_spec("faiss") is not None
    return [
        {"name": "pq_packed", "available": True,
         "trainer": "numpy", "native_runner": True},
        {"name": "pq_legacy", "available": True,
         "requires_artifact": True, "native_runner": True},
        {"name": "pq_qjl_legacy", "available": True,
         "requires_artifact": True, "native_runner": True},
        {"name": "opq", "available": False,
         "dependency_detected": faiss,
         "reason": "faiss adapter is not compiled"},
        {"name": "prq", "available": False,
         "dependency_detected": faiss,
         "reason": "faiss adapter is not compiled"},
        {"name": "jq", "available": False,
         "reason": "JQ source/artifact adapter has not been imported"},
        {"name": "rabitq", "available": False,
         "reason": "optional RaBitQ dependency is not compiled"},
        {"name": "saq", "available": False,
         "reason": "optional SAQ dependency/required ISA is unavailable"},
    ]


def environment_capabilities() -> dict[str, Any]:
    try:
        import numpy
        numpy_version: str | None = numpy.__version__
    except ImportError:
        numpy_version = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": numpy_version,
        "backends": backend_capabilities(),
    }


def require_training_backend(name: str) -> None:
    capability = next((item for item in backend_capabilities()
                       if item["name"] == name), None)
    if capability is None:
        raise ValueError(f"unknown backend: {name}")
    if not capability["available"]:
        raise RuntimeError(f"backend unavailable: {name}: {capability['reason']}")
