$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { "python" }
Set-Location -LiteralPath $RepoRoot
& $Python "./scripts/configure_inhand_reorient_viser.py" @args
exit $LASTEXITCODE
