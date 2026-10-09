"""Build/source/runtime provenance. Formal measurements require exclusive Slurm nodes."""
from __future__ import annotations
import os, platform, subprocess, sys
from pathlib import Path
from .common import identity, digest, verify_file
ROOT=Path(__file__).resolve().parents[3]
THREAD_VARS=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","BLIS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS")

def command(args,required=False):
    p=subprocess.run(list(map(str,args)),text=True,capture_output=True,env={**os.environ,"LC_ALL":"C"})
    if required and p.returncode:raise ValueError(p.stderr or p.stdout)
    return p.stdout.strip()

def source_snapshot():
    entries={}
    for directory in ("hnswlib","tools","examples/cpp","scripts/edge_estimation","scripts/datasets","configs/edge_estimation/final_study"):
        root=ROOT/directory
        if not root.exists():continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".h",".hpp",".cpp",".py",".sh",".json") and "__pycache__" not in path.parts:
                entries[path.relative_to(ROOT).as_posix()]=identity(path)["sha256"]
    entries["CMakeLists.txt"]=identity(ROOT/"CMakeLists.txt")["sha256"]
    return {"head":command(["git","-C",ROOT,"rev-parse","HEAD"],True),
            "branch":command(["git","-C",ROOT,"branch","--show-current"],True),
            "files":entries,"content_sha256":digest(entries)}

def runtime(formal=False):
    env={key:os.environ.get(key) for key in THREAD_VARS}
    if any(value!="1" for value in env.values()):raise ValueError("all search/BLAS thread environment values must be 1")
    info={"hostname":platform.node(),"platform":platform.platform(),"machine":platform.machine(),
          "python":sys.version,"thread_environment":env,"slurm_job_id":os.environ.get("SLURM_JOB_ID"),
          "affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None}
    if platform.system()=="Linux":
        info["cpu"]=command(["lscpu"])
        fields=dict(line.split(":",1) for line in info["cpu"].splitlines() if ":" in line)
        info["cpu_contract"]={key:fields.get(key,"").strip() for key in
            ("Architecture","Vendor ID","Model name","CPU family","Model","Stepping","Flags")}
        info["kernel"]=command(["uname","-a"])
        info["governors"]={str(p):p.read_text().strip() for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")}
        info["processes"]=command(["ps","-eo","pid,user,comm,psr,pcpu"])
    if formal:
        if platform.system()!="Linux" or not info["slurm_job_id"]:
            raise ValueError("formal runs require Linux and an exclusive Slurm allocation; use --local-smoke only for tests")
        job=command(["scontrol","show","job",info["slurm_job_id"],"-o"],True)
        if "OverSubscribe=NO" not in job and "Shared=0" not in job:
            raise ValueError("cannot prove exclusive allocation from scontrol")
        if int(os.environ.get("SLURM_JOB_NUM_NODES","0"))!=1:
            raise ValueError("formal worker requires one node")
        info["slurm_job"]=job
    return info

def verify_build(path,formal=False):
    from .common import unseal
    doc=unseal(path)
    for entry in doc["files"].values():verify_file(entry)
    for entry in doc.get("linked_libraries",[]):verify_file(entry)
    now=source_snapshot()
    if now["content_sha256"]!=doc["source"]["content_sha256"]:
        raise ValueError("source differs from compiled build snapshot; rebuild before continuing")
    if formal and doc["profile"]!="performance":raise ValueError("formal runs require capture-OFF performance build")
    return doc
