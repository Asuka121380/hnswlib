param(
    [string]$OutputDir = "",
    [string]$BuildRoot = "",
    [int]$Parallel = 6
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "results\eq_rcp\stage0"
}
if ([string]::IsNullOrWhiteSpace($BuildRoot)) {
    $BuildRoot = Join-Path $RepoRoot "build-eq-rcp-stage0"
}

$CMake = "C:\msys64\ucrt64\bin\cmake.exe"
$CTest = "C:\msys64\ucrt64\bin\ctest.exe"
$Ninja = "C:\msys64\ucrt64\bin\ninja.exe"
$Compiler = "C:\msys64\ucrt64\bin\g++.exe"
$Python = Join-Path $RepoRoot ".venv-msys\bin\python.exe"
foreach ($Required in @($CMake, $CTest, $Ninja, $Compiler, $Python)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required tool not found: $Required"
    }
}

$OffBuild = Join-Path $BuildRoot "off"
$OnBuild = Join-Path $BuildRoot "on"
$OffTrace = Join-Path $OutputDir "off_trace"
$OnTrace = Join-Path $OutputDir "on_trace"
New-Item -ItemType Directory -Force -Path `
    $BuildRoot, $OutputDir, $OffTrace, $OnTrace | Out-Null

function Invoke-Configure([string]$BuildDir, [string]$EqRcp) {
    & $CMake -S $RepoRoot -B $BuildDir -G Ninja `
        -DCMAKE_BUILD_TYPE=Release `
        -DHNSWLIB_EXAMPLES=ON `
        -DHNSWLIB_ENABLE_BASELINE_TRACE=ON `
        -DHNSWLIB_ENABLE_EDGE_QUANT_V0=OFF `
        "-DHNSWLIB_ENABLE_EQ_RCP=$EqRcp" `
        -DHNSWLIB_ENABLE_EQ_RCP_SHADOW=OFF `
        -DHNSWLIB_ENABLE_EQ_RCP_REAL_PRUNING=OFF `
        -DHNSWLIB_ENABLE_EQ_RCP_FINE_TIMING=OFF `
        -DHNSWLIB_BUILD_EQ_RCP_TOOLS=OFF `
        "-DCMAKE_CXX_COMPILER=$Compiler" `
        "-DCMAKE_MAKE_PROGRAM=$Ninja"
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed for $BuildDir"
    }
}

Invoke-Configure $OffBuild "OFF"
Invoke-Configure $OnBuild "ON"

function Assert-CacheValue(
    [string]$BuildDir,
    [string]$Name,
    [string]$Expected
) {
    $Cache = Get-Content -LiteralPath (Join-Path $BuildDir "CMakeCache.txt")
    $Pattern = "^" + [regex]::Escape($Name) + ":BOOL=" +
        [regex]::Escape($Expected) + "$"
    if (-not ($Cache | Select-String -Pattern $Pattern -Quiet)) {
        throw "Unexpected CMake cache value: $Name must be $Expected in $BuildDir"
    }
}

Assert-CacheValue $OffBuild "HNSWLIB_ENABLE_EQ_RCP" "OFF"
Assert-CacheValue $OnBuild "HNSWLIB_ENABLE_EQ_RCP" "ON"

& $CMake --build $OffBuild `
    --target baseline_trace_runner baseline_trace_test eq_rcp_semantic_probe `
    --parallel $Parallel
if ($LASTEXITCODE -ne 0) { throw "Feature-off build failed" }

& $CMake --build $OnBuild `
    --target baseline_trace_runner baseline_trace_test `
             eq_rcp_semantic_probe eq_rcp_feature_isolation_test `
    --parallel $Parallel
if ($LASTEXITCODE -ne 0) { throw "Skeleton-on build failed" }

foreach ($Executable in @(
    (Join-Path $OffBuild "baseline_trace_test.exe"),
    (Join-Path $OnBuild "baseline_trace_test.exe"),
    (Join-Path $OnBuild "eq_rcp_feature_isolation_test.exe")
)) {
    & $Executable
    if ($LASTEXITCODE -ne 0) { throw "Test failed: $Executable" }
}

& $CTest --test-dir $OffBuild --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "Feature-off CTest failed" }
& $CTest --test-dir $OnBuild --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "Skeleton-on CTest failed" }

$OffProbe = Join-Path $OutputDir "semantic_probe_off.txt"
$OnProbe = Join-Path $OutputDir "semantic_probe_on.txt"
& (Join-Path $OffBuild "eq_rcp_semantic_probe.exe") |
    Set-Content -LiteralPath $OffProbe -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw "Feature-off semantic probe failed" }
& (Join-Path $OnBuild "eq_rcp_semantic_probe.exe") |
    Set-Content -LiteralPath $OnProbe -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw "Skeleton-on semantic probe failed" }

function Invoke-Trace([string]$Executable, [string]$TraceDir) {
    & $Executable `
        --output-dir $TraceDir `
        --n-base 512 `
        --n-query 16 `
        --dim 64 `
        --k 10 `
        --ef-search 50 `
        --M 16 `
        --ef-construction 100 `
        --seed 42 `
        --dco-sample-modulus 1
    if ($LASTEXITCODE -ne 0) { throw "Trace runner failed: $Executable" }
}

Invoke-Trace (Join-Path $OffBuild "baseline_trace_runner.exe") $OffTrace
Invoke-Trace (Join-Path $OnBuild "baseline_trace_runner.exe") $OnTrace

& $Python (Join-Path $PSScriptRoot "compare_semantic_outputs.py") `
    --off-probe $OffProbe `
    --on-probe $OnProbe `
    --off-trace-dir $OffTrace `
    --on-trace-dir $OnTrace `
    --output (Join-Path $OutputDir "semantic_comparison.json")
if ($LASTEXITCODE -ne 0) { throw "Semantic comparison failed" }

& $Python (Join-Path $RepoRoot "tests\python\eq_rcp_semantic_comparison_test.py")
if ($LASTEXITCODE -ne 0) { throw "Semantic comparison unit test failed" }

function Expect-ConfigureFailure(
    [string]$Name,
    [string[]]$Flags
) {
    $BuildDir = Join-Path $BuildRoot ("invalid-" + $Name)
    $Log = Join-Path $OutputDir ("invalid_" + $Name + ".log")
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        # A non-zero native exit and stderr are the expected result here.
        # Windows PowerShell otherwise promotes the stderr record to a
        # terminating error before the exit code can be inspected.
        $ErrorActionPreference = "Continue"
        & $CMake -S $RepoRoot -B $BuildDir -G Ninja `
            -DHNSWLIB_EXAMPLES=OFF `
            "-DCMAKE_CXX_COMPILER=$Compiler" `
            "-DCMAKE_MAKE_PROGRAM=$Ninja" `
            @Flags *> $Log
        $ConfigureExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ConfigureExitCode -eq 0) {
        throw "Invalid configuration unexpectedly succeeded: $Name"
    }
}

Expect-ConfigureFailure "shadow_without_root" @(
    "-DHNSWLIB_ENABLE_EQ_RCP=OFF",
    "-DHNSWLIB_ENABLE_EQ_RCP_SHADOW=ON"
)
Expect-ConfigureFailure "real_without_root" @(
    "-DHNSWLIB_ENABLE_EQ_RCP=OFF",
    "-DHNSWLIB_ENABLE_EQ_RCP_REAL_PRUNING=ON"
)
Expect-ConfigureFailure "timing_without_root" @(
    "-DHNSWLIB_ENABLE_EQ_RCP=OFF",
    "-DHNSWLIB_ENABLE_EQ_RCP_FINE_TIMING=ON"
)
Expect-ConfigureFailure "tools_without_root" @(
    "-DHNSWLIB_ENABLE_EQ_RCP=OFF",
    "-DHNSWLIB_BUILD_EQ_RCP_TOOLS=ON"
)
Expect-ConfigureFailure "eq_real_with_shadow" @(
    "-DHNSWLIB_ENABLE_EQ_RCP=ON",
    "-DHNSWLIB_ENABLE_EQ_RCP_SHADOW=ON",
    "-DHNSWLIB_ENABLE_EQ_RCP_REAL_PRUNING=ON"
)
Expect-ConfigureFailure "eq_real_with_v0_real" @(
    "-DHNSWLIB_ENABLE_EQ_RCP=ON",
    "-DHNSWLIB_ENABLE_EQ_RCP_REAL_PRUNING=ON",
    "-DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON",
    "-DHNSWLIB_ENABLE_V0_REAL_PRUNING=ON"
)
Expect-ConfigureFailure "eq_real_with_ratio_real" @(
    "-DHNSWLIB_ENABLE_EQ_RCP=ON",
    "-DHNSWLIB_ENABLE_EQ_RCP_REAL_PRUNING=ON",
    "-DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON",
    "-DHNSWLIB_ENABLE_V0_RATIO_ESTIMATOR=ON",
    "-DHNSWLIB_ENABLE_V0_RATIO_REAL_PRUNING=ON"
)

$SafeRepo = $RepoRoot.Replace('\', '/')
$GitCommit = (& git -c "safe.directory=$SafeRepo" -C $RepoRoot rev-parse HEAD).Trim()
$GitBranch = (& git -c "safe.directory=$SafeRepo" -C $RepoRoot branch --show-current).Trim()
$GitStatus = @(& git -c "safe.directory=$SafeRepo" -C $RepoRoot status --porcelain)
$Manifest = [ordered]@{
    format = "eq_rcp_stage0_validation_manifest_v1"
    status = "PASS"
    branch = $GitBranch
    commit = $GitCommit
    working_tree_dirty = ($GitStatus.Count -gt 0)
    off_build = $OffBuild
    on_build = $OnBuild
    semantic_comparison = (Join-Path $OutputDir "semantic_comparison.json")
    invalid_configurations = @(
        "shadow_without_root",
        "real_without_root",
        "timing_without_root",
        "tools_without_root",
        "eq_real_with_shadow",
        "eq_real_with_v0_real",
        "eq_real_with_ratio_real"
    )
}
$Manifest | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $OutputDir "stage0_manifest.json") -Encoding UTF8

Write-Output "eq_rcp_stage0_validation_ok"
Write-Output "output_dir=$OutputDir"
