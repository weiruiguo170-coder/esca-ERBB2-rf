param(
    [ValidateSet("demo", "preprocess", "bootstrap", "train_rf", "baselines", "importance")]
    [string]$Mode = "demo",
    [string]$Config = "configs/config.demo.yaml",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    if ($env:PYTHON_EXE) {
        $PythonExe = $env:PYTHON_EXE
    } else {
        $PythonExe = "python"
    }
}

Write-Host "[INFO] RepoRoot: $RepoRoot"
Write-Host "[INFO] Python:   $PythonExe"
Write-Host "[INFO] Mode:     $Mode"
Write-Host "[INFO] Config:   $Config"

& $PythonExe ".\scripts\run_reproduce.py" --mode $Mode --config $Config
Write-Host "[DONE] Pipeline mode '$Mode' finished."

