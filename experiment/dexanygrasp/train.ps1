$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { "python" }

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Could not find Python at $Python"
    exit 1
}

Set-Location -LiteralPath $RepoRoot
Write-Host "Running from `"$($PWD.ProviderPath)`""
Write-Host "Add --debug true to run 9 envs with the Isaac Sim viewer."

$CommandArgs = @(
    ".\scripts\train_dexanygrasp.py",
    "--num_envs", "8192",
    "--max_iterations", "50000",
    "--run_name", "ducky_grasp"
) + $args

Write-Host "`"$Python`" $($CommandArgs -join ' ')"
& $Python @CommandArgs
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    Write-Host
    Write-Host "experiment\dexanygrasp\train.ps1 failed with exit code $ExitCode."
    Read-Host "Press Enter to continue" | Out-Null
}
exit $ExitCode
