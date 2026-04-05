# 数据目录说明（严格版）

本目录不包含任何真实数据文件。

## 保留内容

- `example_input_template.csv`：输入模板，仅表头，不含真实样本行。
- `data_dictionary.md`：字段含义说明。
- `raw/README.md`、`input/README.md`、`local/README.md`、`sample/README.md`：目录用途说明。

## 不保留内容

- 任何真实原始数据或疑似真实数据（csv/tsv/xlsx/xls/parquet/feather/pkl/joblib/h5/hdf5）。
- 任何中间结果、模型输出、训练产物。

## 使用方式

1. 使用 `example_input_template.csv` 作为字段模板准备你自己的数据。
2. 将你的输入文件放到 `data/input/`（本地使用，不提交）。
3. 按 `configs/config.example.yaml` 或 `configs/config.demo.yaml` 运行脚本。

