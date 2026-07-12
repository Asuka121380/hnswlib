# Minimal Pipeline Environment

This repository uses the MSYS2 UCRT64 toolchain for local Windows pipeline validation.

## Installed tools

- GCC 16.1.0
- CMake 4.3.4
- Ninja 1.13.2
- Python 3.14.6
- Python packages pinned in `requirements-pipeline.txt`

The project environment is located at:

```text
.venv-msys\bin\python.exe
```

It was created with `--system-site-packages` and uses the precompiled MSYS2 UCRT64 Python packages.

## PowerShell usage

The current Windows execution policy blocks `Activate.ps1`. Do not change the global policy just for this project. Invoke the environment's Python explicitly:

```powershell
$Python = ".\.venv-msys\bin\python.exe"
& $Python --version
& $Python -m pytest --version
```

If activation is desired for one temporary PowerShell process, start that process with an execution-policy override rather than changing the user or machine policy.

## Configure and build

From the repository root:

```powershell
cmake -S . -B build-local -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=C:\msys64\ucrt64\bin\g++.exe `
  -DCMAKE_MAKE_PROGRAM=C:\msys64\ucrt64\bin\ninja.exe
cmake --build build-local --parallel 6
```

The repository CMake configuration keeps `-lrt` for GNU/Linux and omits it for MinGW, where `librt` does not exist.

## Verified capabilities

The local environment has been verified for:

- building all current CMake targets with GCC and Ninja;
- standard HNSW search tests, filter search, epsilon search and multivector search;
- multithreaded load/update tests;
- NumPy/Pandas processing;
- Parquet write/read through PyArrow;
- PCA through scikit-learn;
- correlation statistics through SciPy;
- headless Matplotlib plot generation;
- pytest execution.

## End-to-end validation

Run the complete high-dimensional, small-data validation pipeline from any PowerShell session:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\baseline_trace\run_pipeline_validation.ps1
```

The default fixture uses `N=2000`, `D=1024`, 32 queries, `k=10`, and `efSearch=100`. It builds the trace-enabled runner, executes the C++ consistency test, compares tracing-off and tracing-on search results, generates exact ground truth, writes query/DCO/edge files, converts trace tables to Parquet, runs PCA/statistical validation, and creates the required plots.

Important parameters can be overridden without editing the script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\baseline_trace\run_pipeline_validation.ps1 `
  -NBase 5000 -NQuery 100 -Dimension 1536 `
  -K 10 -EfSearch 200 -DcoSampleModulus 5
```

Outputs are written under `results/pipeline_validation/` and local build files under `build-pipeline/`; both directories are ignored by Git.

For headless or restricted runs, set a writable Matplotlib cache directory:

```powershell
$env:MPLCONFIGDIR = Join-Path $env:TEMP "hnsw-matplotlib"
```

## Server environment

Use a supported server Python version and install:

```bash
python -m pip install -r requirements-pipeline.txt
```

The server compiler and library versions do not need to match Windows exactly, but the trace schema, Python package versions, random seeds, CMake build type and git commit must be recorded in each run's metadata.
