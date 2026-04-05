# scripts 说明

当前建议使用以下分步脚本：

1. `01_preprocess.py`：构建联合终点、7:3 划分、仅训练集拟合预处理
2. `02_bootstrap.py`：训练集内 Bootstrap 到 480
3. `03_train_rf.py`：随机森林训练与参数选择（CV R² + OOB R²）
4. `04_baselines.py`：线性回归与 SVR(RBF) 对照
5. `05_feature_importance.py`：杂质重要性 + 置换重要性 + Spearman 聚类代表特征
6. `06_demo_run.py`：一键串行执行 01-05（轻量 demo）

快速命令：

```bash
python scripts/06_demo_run.py --config configs/config.demo.yaml --force-demo-data
```

说明：

- 默认不触发大规模训练。
- 若需完整参数搜索，使用 `03_train_rf.py --strategy grid_search`。

