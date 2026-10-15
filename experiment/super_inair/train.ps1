$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { "python" }
Set-Location -LiteralPath $RepoRoot
& $Python "./scripts/train_super_inair.py" --num_envs 8192 --max_iterations 100000 --max-training-hours 1 --run_name tennis_fixed_base_1h @args
exit $LASTEXITCODE
