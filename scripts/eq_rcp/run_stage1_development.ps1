param(
    [string]$Config = "configs/eq_rcp/stage1_development_gist1m.json",
    [string]$OutputRoot = "results/eq_rcp/stage1/development"
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
$runRoot = Join-Path $rootPath $runId
$outputA = Join-Path $runRoot "materialization-a"
$outputB = Join-Path $runRoot "materialization-b"

$env:PYTHONDONTWRITEBYTECODE = "1"
& $python (Join-Path $repoRoot "tests/python/eq_rcp_dataset_test.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python (Join-Path $PSScriptRoot "build_operational_dataset.py") --config $configPath --output-dir $outputA
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python (Join-Path $PSScriptRoot "validate_operational_dataset.py") --dataset-dir $outputA --geometry-record-limit 0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python (Join-Path $PSScriptRoot "build_operational_dataset.py") --config $configPath --output-dir $outputB
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python (Join-Path $PSScriptRoot "validate_operational_dataset.py") --dataset-dir $outputB --geometry-record-limit 4096
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$manifestA = Get-Content -LiteralPath (Join-Path $outputA "dataset_manifest.json") -Raw | ConvertFrom-Json
$manifestB = Get-Content -LiteralPath (Join-Path $outputB "dataset_manifest.json") -Raw | ConvertFrom-Json
if ($manifestA.dataset_semantic_sha256 -ne $manifestB.dataset_semantic_sha256) {
    throw "Repeatability failure: semantic dataset hashes differ"
}
foreach ($table in @("queries", "edges", "events")) {
    if ($manifestA.outputs.$table.semantic_sha256 -ne $manifestB.outputs.$table.semantic_sha256) {
        throw "Repeatability failure: $table semantic hashes differ"
    }
}

[ordered]@{
    status = "PASS"
    implementation = "PASS"
    local_development_materialization = "PASS"
    cluster_formal_materialization = "PENDING"
    semantic_sha256 = $manifestA.dataset_semantic_sha256
    materialization_a = $outputA
    materialization_b = $outputB
} | ConvertTo-Json | Write-Output
