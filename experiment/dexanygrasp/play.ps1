$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { "python" }

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Could not find Python at $Python"
    exit 1
}

Set-Location -LiteralPath $RepoRoot

$CommandArgs = @(
    ".\scripts\play_dexanygrasp.py"
) + $args

Write-Host "`"$Python`" $($CommandArgs -join ' ')"
& $Python @CommandArgs
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    Write-Host
    Write-Host "experiment\dexanygrasp\play.ps1 failed with exit code $ExitCode."
    Read-Host "Press Enter to continue" | Out-Null
}
exit $ExitCode
