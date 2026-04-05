# GitHub 上传前检查清单

## 一、应上传内容（推荐）
- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `metadata/`
- `data/raw/README.md`
- `data/processed/`（仅轻量可公开文件）
- `results/`
- `figures/`
- `README.md`
- `requirements.txt`
- `environment.yml`
- `.gitignore`
- `LICENSE`
- `CITATION.cff`

## 二、不应上传内容
- 原始大体积数据文件（CCLE/GDSC 原文件）
- 超大中间文件（如 `intermediate/bootstrap_dataset.csv`）
- 本地缓存/日志/IDE 文件

## 三、上传前人工检查
1. `README.md` 是否完整可读  
2. `docs/data_download_instructions.md` 是否清楚  
3. `configs/default.yaml` 的文件名是否与本地一致  
4. `CITATION.cff` 仓库地址与作者信息是否已替换  
5. `LICENSE` 作者信息是否已替换  
6. `results/` 与 `figures/` 是否为你希望公开的版本

## 四、仓库名称建议
- `esca-erbb2-her2-sensitivity-repro`
- `esophageal-erbb2-drug-sensitivity`
- `ccle-gdsc-erbb2-reproduction`

## 五、仓库描述建议
“基于 CCLE/GDSC 公开数据的食管癌 ERBB2/HER2 靶向药物敏感性预测复现仓库，包含数据处理、建模、评估与论文图表生成流程。”
