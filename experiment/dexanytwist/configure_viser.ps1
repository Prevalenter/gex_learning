$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { "python" }

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Could not find Python at $Python"
    exit 1
}

Set-Location -LiteralPath $RepoRoot
Write-Host "Running from `"$($PWD.ProviderPath)`""
Write-Host "Opening the lightweight DexAnyTwist viser configurator."
Write-Host "Install once with: $Python -m pip install -e `".[viser]`""

$CommandArgs = @(
    ".\scripts\configure_dexanytwist_viser.py",
    "--object-name", "bulb_custom_0017"
) + $args

Write-Host "`"$Python`" $($CommandArgs -join ' ')"
& $Python @CommandArgs
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    Write-Host
    Write-Host "experiment\dexanytwist\configure_viser.ps1 failed with exit code $ExitCode."
    Read-Host "Press Enter to continue" | Out-Null
}
exit $ExitCode
