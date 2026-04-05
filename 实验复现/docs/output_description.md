# 输出结果说明

默认输出目录为 `outputs/demo/`（配置可改）。

## 1. 预处理与抽样

- `processed/train_processed.csv`：训练集（特征已按训练集统计量标准化）
- `processed/test_processed.csv`：测试集（使用训练集标准化器变换）
- `processed/train_bootstrap.csv`：训练集 Bootstrap 扩增到 480 条后的数据
- `processed/preprocess_meta.json`：划分比例、随机种子、终点定义、样本量等元信息

## 2. 模型性能

- `metrics/rf_metrics.json`：RF 最优参数、CV R²、OOB R²、测试集指标
- `metrics/rf_selection_scores.csv`：参数组合评分表（CV R² + OOB R²）
- `metrics/baseline_metrics.json`：RF/LR/SVR 的对比指标
- `metrics/predictions_demo.csv`：测试集真实值与各模型预测值

## 3. 特征解释

- `feature_importance/feature_importance_demo.csv`：杂质重要性 + 置换重要性 + 综合分数
- `feature_importance/feature_clusters_demo.csv`：Spearman 聚类簇成员
- `feature_importance/representative_features_demo.csv`：每簇代表特征

## 4. 图形与摘要

- `figures/prediction_scatter_demo.png`：预测散点图
- `figures/feature_importance_demo.png`：Top10 综合重要性条形图
- `run_summary.md`：本次运行摘要

## 5. 说明

- 以上 demo 输出仅用于流程验证与接口演示，不代表论文正式结果。
- 正式结果需要替换为真实整理数据并在完整计算条件下运行。

