# 投稿图表一键构建脚本：
# - 先生成 submission 图表/表格
# - 再生成图4生物学约束敏感性分析
param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    if ($env:PYTHON_EXE) {
        $PythonExe = $env:PYTHON_EXE
    } elseif (Test-Path "C:\Users\18210\AppData\Local\Programs\Python\Python313\python.exe") {
        $PythonExe = "C:\Users\18210\AppData\Local\Programs\Python\Python313\python.exe"
    } else {
        $PythonExe = "python"
    }
}

Write-Host "[INFO] Build submission figures/tables..."
& $PythonExe ".\src\project_cli.py" --mode submission --python $PythonExe
& $PythonExe ".\src\project_cli.py" --mode figure4_sensitivity --python $PythonExe
Write-Host "[DONE] Submission assets generated."
