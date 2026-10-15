$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = if ($env:PYTHON_EXECUTABLE) { $env:PYTHON_EXECUTABLE } else { "python" }

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Could not find Python at $Python"
    exit 1
}

Set-Location -LiteralPath $RepoRoot
Write-Host "Running from `"$($PWD.ProviderPath)`""
Write-Host "Play opens the Isaac Sim viewer by default. Add --headless to run without the viewer."
Write-Host "Default play uses 9 envs. Add --num_envs N to override."
Write-Host "Add --debug true to show keypoint markers and force 9 envs."
Write-Host "Add --save-hand-trajectory true to save hand trajectory NPZ chunks."

function Convert-ToBoolFlag {
    param(
        [string]$Value,
        [string]$FlagName
    )
    $Normalized = $Value.Trim().ToLowerInvariant()
    switch ($Normalized) {
        { $_ -in @("1", "true", "yes", "on") } { return $true }
        { $_ -in @("0", "false", "no", "off") } { return $false }
        default { throw "Invalid value for ${FlagName}: $Value. Use true or false." }
    }
}

function Test-HydraOverride {
    param(
        [object[]]$ArgList,
        [string]$Key
    )
    foreach ($Arg in $ArgList) {
        $Text = [string]$Arg
        if ($Text.TrimStart("+").StartsWith("$Key=")) {
            return $true
        }
    }
    return $false
}

function Test-Option {
    param(
        [object[]]$ArgList,
        [string[]]$Names
    )
    foreach ($Arg in $ArgList) {
        $Text = [string]$Arg
        foreach ($Name in $Names) {
            if ($Text -eq $Name -or $Text.StartsWith("$Name=")) {
                return $true
            }
        }
    }
    return $false
}

$SaveHandTrajectory = $false
$ForwardArgs = @()
$SaveHandTrajectoryPrefix = "--save-hand-trajectory="
for ($Index = 0; $Index -lt $args.Count; $Index++) {
    $ArgText = [string]$args[$Index]
    if ($ArgText -eq "--save-hand-trajectory") {
        if ($Index + 1 -lt $args.Count) {
            $NextText = [string]$args[$Index + 1]
            if (-not $NextText.StartsWith("--")) {
                $SaveHandTrajectory = Convert-ToBoolFlag $NextText "--save-hand-trajectory"
                $Index += 1
                continue
            }
        }
        $SaveHandTrajectory = $true
        continue
    }
    if ($ArgText.StartsWith($SaveHandTrajectoryPrefix)) {
        $SaveHandTrajectory = Convert-ToBoolFlag ($ArgText.Substring($SaveHandTrajectoryPrefix.Length)) "--save-hand-trajectory"
        continue
    }
    $ForwardArgs += $args[$Index]
}

$TrajectoryHydraArgs = @()
if ($SaveHandTrajectory) {
    $TrajectoryHydraArgs += "env.save_hand_trajectory=True"
    $TrajectoryHydraArgs += "env.hand_trajectory_dir=experiment/dexanytwist/hand_trajectories"
    $TrajectoryHydraArgs += "env.hand_trajectory_interval=1000"
}

$ScaleRangeHydraArgs = @()
if (-not (Test-HydraOverride $ForwardArgs "env.enable_scale_range")) {
    $ScaleRangeHydraArgs += "env.enable_scale_range=True"
}

$NumEnvArgs = @()
if (-not (Test-Option $ForwardArgs @("--num_envs"))) {
    $NumEnvArgs += "--num_envs"
    $NumEnvArgs += "9"
}

$CommandArgs = @(
    ".\scripts\play_dexanytwist.py",
    "--object-name", "bulb_custom_0017"
) + $NumEnvArgs + $ScaleRangeHydraArgs + $TrajectoryHydraArgs + $ForwardArgs

Write-Host "`"$Python`" $($CommandArgs -join ' ')"
& $Python @CommandArgs
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    Write-Host
    Write-Host "experiment\dexanytwist\play.ps1 failed with exit code $ExitCode."
    Read-Host "Press Enter to continue" | Out-Null
}
exit $ExitCode
