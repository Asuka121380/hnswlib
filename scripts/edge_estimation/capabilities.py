from __future__ import annotations

import importlib.util
import platform
from typing import Any


def backend_capabilities() -> list[dict[str, Any]]:
    faiss = importlib.util.find_spec("faiss") is not None
    return [
        {"name": "pq_packed", "trainer_available": True,
         "trainer_providers": ["numpy_reference"] + (["faiss"] if faiss else []),
         "artifact_supported": True},
        {"name": "pq_legacy", "trainer_available": False,
         "artifact_supported": True,
         "reason": "native build option and frozen legacy sidecar required"},
        {"name": "pq_qjl_legacy", "trainer_available": False,
         "artifact_supported": True,
         "reason": "native build option and frozen sidecar/QJL companion required"},
        {"name": "opq", "trainer_available": faiss,
         "trainer_providers": ["faiss"] if faiss else [],
         "artifact_supported": True,
         "reason": None if faiss else "faiss trainer module is unavailable"},
        {"name": "prq", "trainer_available": faiss,
         "trainer_providers": ["faiss"] if faiss else [],
         "dependency_detected": faiss, "artifact_supported": True,
         "reason": None if faiss else "faiss trainer module is unavailable"},
        {"name": "jq", "trainer_available": True,
         "trainer_providers": ["behavioral_port"], "artifact_supported": True,
         "reason": "behavioral port; author-binary oracle pending"},
        {"name": "rabitq", "trainer_available": faiss,
         "trainer_providers": ["faiss"] if faiss else [],
         "dependency_detected": faiss, "artifact_supported": True,
         "reason": None if faiss else "faiss trainer module is unavailable"},
        {"name": "saq", "trainer_available": False, "artifact_supported": False,
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
    if not capability["trainer_available"]:
        raise RuntimeError(f"backend unavailable: {name}: {capability['reason']}")
