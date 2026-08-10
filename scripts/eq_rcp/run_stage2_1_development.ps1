param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python (Join-Path $scriptDir "run_stage2_1_selection.py") --config $Config --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python (Join-Path $scriptDir "validate_stage2_1_selection.py") --config $Config --result-dir $OutputDir
exit $LASTEXITCODE
