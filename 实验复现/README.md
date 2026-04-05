# 实验复现（GitHub 安全发布版）

本仓库仅保留代码、配置、文档、输入模板。  
不包含任何真实原始数据、真实中间结果、真实训练输出。

## 数据发布原则

- 不发布 GDSC/CCLE 原始数据文件。
- 不发布任何疑似真实数据文件（包括 csv/xlsx/tsv/parquet/pkl/joblib/h5 等）。
- 当前只提供：
  - `data/example_input_template.csv`（仅表头模板）
  - `data/data_dictionary.md`（字段说明）
  - `data/*/README.md`（目录占位说明）

## 你需要自行准备的数据

1. 从 GDSC / CCLE 获取原始数据。
2. 整理为与模板一致字段的输入 CSV。
3. 放到本地 `data/input/`（该目录中的数据文件默认不会被跟踪）。

## 轻量 demo 说明

- `scripts/06_demo_run.py` 会在本地自动生成临时 demo CSV 到 `data/local/`，仅用于流程联通验证。
- 该本地生成文件不应提交到 GitHub。

## 运行

```bash
pip install -r requirements.txt
python scripts/06_demo_run.py --config configs/config.demo.yaml --force-demo-data
```

## 相关说明

- 数据说明：`docs/data_description.md`
- 输入字段说明：`data/data_dictionary.md`

