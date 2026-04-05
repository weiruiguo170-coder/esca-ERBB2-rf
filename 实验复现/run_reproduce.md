# 运行命令速查

## 轻量 demo（推荐）

```bash
python scripts/06_demo_run.py --config configs/config.demo.yaml --force-demo-data
```

## 分步运行

```bash
python scripts/01_preprocess.py --config configs/config.example.yaml
python scripts/02_bootstrap.py --config configs/config.example.yaml
python scripts/03_train_rf.py --config configs/config.example.yaml --strategy grid_search
python scripts/04_baselines.py --config configs/config.example.yaml
python scripts/05_feature_importance.py --config configs/config.example.yaml --mode full
```

## 注意

- 默认配置不执行长时间全量训练。
- 正式复现请先用真实整理数据替换 `paths.input_csv`。

