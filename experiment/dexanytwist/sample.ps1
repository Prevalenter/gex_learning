$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { "python" }

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Could not find Python at $Python"
    exit 1
}

function Test-Option([object[]]$ArgList, [string]$Name) {
    foreach ($Arg in $ArgList) {
        $Text = [string]$Arg
        if ($Text -eq $Name -or $Text.StartsWith("$Name=")) { return $true }
    }
    return $false
}

function Test-HydraOverride([object[]]$ArgList, [string]$Key) {
    foreach ($Arg in $ArgList) {
        if (([string]$Arg).TrimStart("+").StartsWith("$Key=")) { return $true }
    }
    return $false
}

$Defaults = @()
if (-not (Test-Option $args "--num_envs")) { $Defaults += @("--num_envs", "128") }
if (-not (Test-HydraOverride $args "env.enable_scale_range")) { $Defaults += "env.enable_scale_range=False" }
if (-not (Test-HydraOverride $args "env.scene.replicate_physics")) { $Defaults += "env.scene.replicate_physics=True" }
if (-not (Test-HydraOverride $args "env.scene.clone_in_fabric")) { $Defaults += "env.scene.clone_in_fabric=True" }
if (-not (Test-HydraOverride $args "env.save_hand_trajectory")) { $Defaults += "env.save_hand_trajectory=True" }
if (-not (Test-HydraOverride $args "env.hand_trajectory_dir")) {
    $Defaults += "env.hand_trajectory_dir=experiment/hand_trajectories"
}
if (-not (Test-HydraOverride $args "env.hand_trajectory_interval")) {
    $Defaults += "env.hand_trajectory_interval=1000"
}
if (-not (Test-HydraOverride $args "env.hand_trajectory_max_steps")) {
    $Defaults += "env.hand_trajectory_max_steps=1000"
}
if (-not (Test-HydraOverride $args "env.hand_trajectory_quit_on_complete")) {
    $Defaults += "env.hand_trajectory_quit_on_complete=True"
}

Set-Location -LiteralPath $RepoRoot
$CommandArgs = @(".\scripts\play_dexanytwist.py", "--object-name", "bulb_custom_0017") + $Defaults + $args
Write-Host "`"$Python`" $($CommandArgs -join ' ')"
& $Python @CommandArgs
exit $LASTEXITCODE
