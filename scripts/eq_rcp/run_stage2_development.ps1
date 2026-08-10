param(
    [string]$Config = "configs/eq_rcp/stage2_development_gist1m.json",
    [string]$OutputRoot = "results/eq_rcp/stage2/development"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$python = Join-Path $repoRoot ".venv-msys/bin/python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $repoRoot $Config }
$rootPath = if ([System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot } else { Join-Path $repoRoot $OutputRoot }
$runId = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$outputDir = Join-Path $rootPath $runId

$env:PYTHONDONTWRITEBYTECODE = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

& $python (Join-Path $repoRoot "tests/python/eq_rcp_stage2_test.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python (Join-Path $PSScriptRoot "run_stage2_ceiling.py") --config $configPath --output-dir $outputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python (Join-Path $PSScriptRoot "validate_stage2_ceiling.py") --config $configPath --result-dir $outputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$manifest = Get-Content -LiteralPath (Join-Path $outputDir "stage2_manifest.json") -Raw | ConvertFrom-Json
[ordered]@{
    status = "PASS"
    implementation = "PASS"
    local_development_ceiling = "PASS"
    formal_stage2_gate = $manifest.gate_decision
    results_semantic_sha256 = $manifest.stage2_results_semantic_sha256
    output_dir = $outputDir
} | ConvertTo-Json | Write-Output
