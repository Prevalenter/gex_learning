$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { "python" }
Set-Location -LiteralPath $RepoRoot
& $Python "./scripts/play_super_inair.py" @args
exit $LASTEXITCODE
