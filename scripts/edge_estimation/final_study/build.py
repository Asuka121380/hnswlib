"""Create and seal a common compiler profile for all three methods."""
from __future__ import annotations
import argparse, os, re, shlex, subprocess
from pathlib import Path
from .common import cli, load, seal, identity, digest
from .environment import ROOT,source_snapshot,command

def build(args):
    dest=Path(args.out).resolve();before=source_snapshot()
    flags={"CMAKE_BUILD_TYPE":"Release","CMAKE_EXPORT_COMPILE_COMMANDS":"ON",
      "HNSWLIB_ENABLE_NATIVE_ARCH":"OFF","HNSWLIB_PERFORMANCE_COMPARABLE_FLAGS":"ON",
      "HNSWLIB_ENABLE_V0_STRICT_FP_CONTRACT":"ON","HNSWLIB_ENABLE_EDGE_QUANT_V0":"ON",
      "HNSWLIB_BUILD_EDGE_ESTIMATION_TOOLS":"ON","HNSWLIB_ENABLE_EDGE_ESTIMATION_ACTIVE":"ON",
      "HNSWLIB_ENABLE_EDGE_ESTIMATION_CAPTURE":"ON" if args.tools else "OFF","UQ_WITH_BLAS":"ON"}
    for option in ("HNSWLIB_ENABLE_BASELINE_TRACE","HNSWLIB_ENABLE_V0_SHADOW_VALIDATION",
                   "HNSWLIB_ENABLE_V0_APPROX_SHADOW","HNSWLIB_ENABLE_V0_REAL_PRUNING",
                   "HNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING","HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR",
                   "HNSWLIB_ENABLE_V0_RESIDUAL_REAL_PRUNING"):
        flags[option]="OFF"
    if args.blas:flags["UQ_BLAS_LIBRARY"]=str(Path(args.blas).resolve(strict=True))
    subprocess.run([args.cmake,"-S",str(ROOT),"-B",str(dest),"-G","Ninja",*[f"-D{k}={v}" for k,v in flags.items()]],check=True)
    targets=["uq_opq_performance_runner","uq_ivf_performance_runner"]
    if args.tools:targets+=["uq_capture","uq_capture_fixture","uq_native_runner","real_data_trace_runner"]
    subprocess.run([args.cmake,"--build",str(dest),"--target",*targets,"--parallel",str(args.jobs)],check=True)
    entries=load(dest/"compile_commands.json")
    cmds=[e.get("command"," ".join(e.get("arguments",[]))) for e in entries if Path(e["file"]).name in ("uq_opq_performance_runner.cpp","uq_ivf_performance_runner.cpp")]
    if len(cmds)!=2:raise ValueError("both common runners must have compile commands")
    for cmd in cmds:
        for required in ("-O3","-std=c++17","-fno-fast-math","-ffp-contract=off","UQ_EDGE_ESTIMATION_HAVE_BLAS=1"):
            if required not in cmd:raise ValueError("missing compiler requirement: "+required)
        if re.search(r"-m(?:arch|cpu)=native|-ffast-math|-Ofast",cmd):raise ValueError("nonportable/unsafe compiler optimization")
        if not args.tools and "HNSWLIB_ENABLE_EDGE_ESTIMATION_CAPTURE=1" in cmd:raise ValueError("capture hook in performance build")
    suffix=".exe" if os.name=="nt" else ""
    files={t:identity(dest/(t+suffix)) for t in targets}
    files.update(compile_commands=identity(dest/"compile_commands.json"),cmake_cache=identity(dest/"CMakeCache.txt"))
    libraries={}
    if os.name!="nt":
        for target in targets[:2]:
            output=command(["ldd",files[target]["path"]],True)
            for match in re.finditer(r"(?:=>\s+)?(/[^\s]+)",output):
                path=Path(match.group(1))
                if path.is_file():libraries[str(path.resolve())]=identity(path)
    elif args.blas:libraries[args.blas]=identity(args.blas)
    after=source_snapshot()
    if before["content_sha256"]!=after["content_sha256"]:raise ValueError("source changed during build")
    seal(dest/"build_manifest.json",{"schema_version":2,"profile":"tools" if args.tools else "performance",
        "flags":flags,"commands":cmds,"source":after,"files":files,"linked_libraries":list(libraries.values()),
        "compiler":command([shlex.split(cmds[0],posix=os.name!="nt")[0].strip('"'),"--version"],True)})
    print(dest/"build_manifest.json")
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",required=True)
    p.add_argument("--cmake",default="cmake");p.add_argument("--blas");p.add_argument("--tools",action="store_true")
    p.add_argument("--jobs",type=int,default=4);build(p.parse_args())
if __name__=="__main__":cli(main)
