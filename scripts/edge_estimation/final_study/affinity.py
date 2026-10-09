"""Pin the Slurm step before importing numerical libraries or starting runners."""
from __future__ import annotations
import json
import os
import sys


def require_single_cpu(cpus, expected=None):
    if (not isinstance(cpus, (list, tuple)) or len(cpus) != 1
            or type(cpus[0]) is not int or cpus[0] < 0):
        raise ValueError("measurement requires affinity to exactly one logical CPU; use worker.sh")
    if expected is not None and list(cpus) != list(expected):
        raise ValueError("measurement CPU affinity changed during the allocation")
    return cpus[0]


def check_current_cpu(expected=None):
    if not hasattr(os, "sched_getaffinity"):
        raise ValueError("CPU affinity inspection is unavailable")
    return require_single_cpu(sorted(os.sched_getaffinity(0)), expected)


def pin_current_process():
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise ValueError("Linux CPU affinity APIs are required")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise ValueError("Slurm step has no allowed CPUs")
    # Select inside the step's allowed set; do not assume CPU 0 is available.
    cpu = allowed[0]
    os.sched_setaffinity(0, {cpu})
    check_current_cpu([cpu])
    print("CPU_PIN=" + json.dumps({"allowed_before": allowed, "selected_cpu": cpu}),
          file=sys.stderr, flush=True)
    return cpu


def main():
    if sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("pinned worker requires Linux and a Slurm allocation")
    pin_current_process()
    # exec starts the driver (and all its library threads) with the verified mask.
    os.execv(sys.executable, [
        sys.executable, "-m", "scripts.edge_estimation.final_study.run", *sys.argv[1:]
    ])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"CPU_PIN_FAILED: {error}") from error
