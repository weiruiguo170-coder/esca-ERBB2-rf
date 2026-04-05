# 文件清单（轻量复现主链）

## 必要文件

- `README.md`
- `requirements.txt`
- `environment.yml`
- `configs/config.demo.yaml`
- `configs/config.example.yaml`
- `src/lightweight_repro_pipeline.py`
- `scripts/01_preprocess.py`
- `scripts/02_bootstrap.py`
- `scripts/03_train_rf.py`
- `scripts/04_baselines.py`
- `scripts/05_feature_importance.py`
- `scripts/06_demo_run.py`
- `data/README.md`
- `docs/how_to_reproduce.md`
- `docs/output_description.md`

## 保留但默认不触发的大任务接口

- 旧版完整脚本（`src/reproduce_core.py` 等）
- 历史论文图表脚本（`src/submission_builder.py` 等）

