param(
    [string]$OutputDir = "",
    [string]$BuildDir = "",
    [int]$NBase = 2000,
    [int]$NQuery = 32,
    [int]$Dimension = 1024,
    [int]$K = 10,
    [int]$EfSearch = 100,
    [int]$M = 16,
    [int]$EfConstruction = 100,
    [int]$Seed = 42,
    [int]$DcoSampleModulus = 1
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "results\pipeline_validation"
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $RepoRoot "build-pipeline"
}

$RawDir = Join-Path $OutputDir "raw"
$AnalysisDir = Join-Path $OutputDir "analysis"
New-Item -ItemType Directory -Force -Path $BuildDir, $RawDir, $AnalysisDir | Out-Null

$CMake = "C:\msys64\ucrt64\bin\cmake.exe"
$Ninja = "C:\msys64\ucrt64\bin\ninja.exe"
$Compiler = "C:\msys64\ucrt64\bin\g++.exe"
$Python = Join-Path $RepoRoot ".venv-msys\bin\python.exe"

foreach ($Required in @($CMake, $Ninja, $Compiler, $Python)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required tool not found: $Required"
    }
}

$env:MPLCONFIGDIR = Join-Path $env:TEMP "hnsw-matplotlib"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& $CMake -S $RepoRoot -B $BuildDir -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DHNSWLIB_ENABLE_BASELINE_TRACE=ON `
    "-DCMAKE_CXX_COMPILER=$Compiler" `
    "-DCMAKE_MAKE_PROGRAM=$Ninja"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

& $CMake --build $BuildDir --target baseline_trace_runner baseline_trace_test --parallel 6
if ($LASTEXITCODE -ne 0) { throw "C++ build failed" }

& (Join-Path $BuildDir "baseline_trace_test.exe")
if ($LASTEXITCODE -ne 0) { throw "baseline_trace_test failed" }

& (Join-Path $BuildDir "baseline_trace_runner.exe") `
    --output-dir $RawDir `
    --n-base $NBase `
    --n-query $NQuery `
    --dim $Dimension `
    --k $K `
    --ef-search $EfSearch `
    --M $M `
    --ef-construction $EfConstruction `
    --seed $Seed `
    --dco-sample-modulus $DcoSampleModulus
if ($LASTEXITCODE -ne 0) { throw "baseline_trace_runner failed" }

& $Python (Join-Path $PSScriptRoot "analyze_trace.py") `
    --input-dir $RawDir `
    --output-dir $AnalysisDir
if ($LASTEXITCODE -ne 0) { throw "Trace analysis failed" }

$Expected = @(
    (Join-Path $RawDir "metadata.json"),
    (Join-Path $RawDir "query_stats.csv"),
    (Join-Path $RawDir "dco_trace.csv"),
    (Join-Path $RawDir "edge_directions.csv"),
    (Join-Path $AnalysisDir "summary.csv"),
    (Join-Path $AnalysisDir "query_stats.parquet"),
    (Join-Path $AnalysisDir "dco_trace.parquet"),
    (Join-Path $AnalysisDir "validation_report.json"),
    (Join-Path $AnalysisDir "report.md"),
    (Join-Path $AnalysisDir "plots\edge_direction_pca.png")
)
foreach ($Artifact in $Expected) {
    if (-not (Test-Path -LiteralPath $Artifact)) {
        throw "Expected artifact was not generated: $Artifact"
    }
}

Write-Output "pipeline_validation_ok"
Write-Output "output_dir=$OutputDir"
