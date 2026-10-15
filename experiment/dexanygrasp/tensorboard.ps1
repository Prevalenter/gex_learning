$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { "python" }
$LogDir = "logs\rsl_rl\gx4m_dexanygrasp_ducky"
$ExtraArgs = @($args)

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Could not find Python at $Python"
    exit 1
}

Set-Location -LiteralPath $RepoRoot

if ($args.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace([string]$args[0]) -and -not ([string]$args[0]).StartsWith("-")) {
    $First = [string]$args[0]
    $LogDir = $First
    $ExperimentLogDir = Join-Path "logs\rsl_rl\gx4m_dexanygrasp_ducky" $First
    if (-not (Test-Path -LiteralPath $LogDir) -and (Test-Path -LiteralPath $ExperimentLogDir)) {
        $LogDir = $ExperimentLogDir
    }
    if ($args.Count -gt 1) {
        $ExtraArgs = @($args[1..($args.Count - 1)])
    } else {
        $ExtraArgs = @()
    }
}

$CommandArgs = @(
    "-m", "tensorboard.main",
    "--logdir", $LogDir,
    "--port", "6006",
    "--host", "127.0.0.1"
) + $ExtraArgs

Write-Host "`"$Python`" $($CommandArgs -join ' ')"
& $Python @CommandArgs
exit $LASTEXITCODE
