# 统一整理结果文件：
# 将历史产物复制到 results/ 与 figures/，不删除原始来源目录文件。
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path ".\figures")) { New-Item -ItemType Directory -Path ".\figures" | Out-Null }
if (-not (Test-Path ".\results")) { New-Item -ItemType Directory -Path ".\results" | Out-Null }
if (-not (Test-Path ".\data\processed")) { New-Item -ItemType Directory -Path ".\data\processed" -Force | Out-Null }

$copyMode = @{}
if ($Force) { $copyMode["Force"] = $true }

$mappings = @(
    @{src="submission_figures_tables\table1_revised_model_performance.csv"; dst="results\table1_model_performance.csv"},
    @{src="submission_figures_tables\table2_revised_top10_features.csv"; dst="results\table2_top10_features.csv"},
    @{src="submission_figures_tables\_artifacts_predictions.csv"; dst="results\figure3_predictions.csv"},
    @{src="submission_figures_tables\_artifacts_top_features.csv"; dst="results\feature_importance_raw_top.csv"},
    @{src="results_quick\metrics_summary.csv"; dst="results\metrics_summary_quick.csv"},
    @{src="results_quick\model_comparison.csv"; dst="results\model_comparison_quick.csv"},
    @{src="submission_figures_tables\figure1_revised.png"; dst="figures\figure1_main.png"},
    @{src="submission_figures_tables\figure1_revised.pdf"; dst="figures\figure1_main.pdf"},
    @{src="submission_figures_tables\figure3_revised.png"; dst="figures\figure3_prediction_performance.png"},
    @{src="submission_figures_tables\figure3_revised.pdf"; dst="figures\figure3_prediction_performance.pdf"},
    @{src="submission_figures_tables\figure4_revised.png"; dst="figures\figure4_feature_importance.png"},
    @{src="submission_figures_tables\figure4_revised.pdf"; dst="figures\figure4_feature_importance.pdf"},
    @{src="figure4_sensitivity_outputs\figure4_biology_constrained_preview.png"; dst="figures\figure4_sensitivity_biology_constrained.png"},
    @{src="figure4_sensitivity_outputs\figure4_comparison.png"; dst="figures\figure4_ranking_comparison.png"},
    @{src="figure4_sensitivity_outputs\figure4_biology_constrained.csv"; dst="results\figure4_biology_constrained.csv"},
    @{src="figure4_sensitivity_outputs\figure4_data_driven_current.csv"; dst="results\figure4_data_driven_current.csv"},
    @{src="figure4_sensitivity_outputs\figure4_biology_constrained_heatmap.csv"; dst="results\figure4_biology_constrained_heatmap.csv"},
    @{src="teaching_figure1_matched_dataset.csv"; dst="data\processed\teaching_figure1_matched_dataset.csv"},
    @{src="teaching_figure3_matched_dataset.csv"; dst="data\processed\teaching_figure3_matched_dataset.csv"},
    @{src="teaching_figure4_matched_dataset.csv"; dst="data\processed\teaching_figure4_matched_dataset.csv"}
)

foreach ($m in $mappings) {
    if (Test-Path $m.src) {
        Copy-Item -Path $m.src -Destination $m.dst @copyMode
        Write-Host "[COPIED] $($m.src) -> $($m.dst)"
    } else {
        Write-Host "[SKIP] missing: $($m.src)"
    }
}

Write-Host "[DONE] Output organization finished."
