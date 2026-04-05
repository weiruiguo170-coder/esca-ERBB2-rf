# How To Reproduce (Strict Data Policy)

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Data preparation

- This repository does not ship any real data.
- Use `data/example_input_template.csv` as the schema template.
- Prepare your own merged input CSV from GDSC/CCLE and place it under `data/input/` (local only).

## 3. Demo run

```bash
python scripts/06_demo_run.py --config configs/config.demo.yaml --force-demo-data
```

The demo script generates temporary local CSV under `data/local/` and does not require committed sample data files.

## 4. Full interface (with your own data)

```bash
python scripts/01_preprocess.py --config configs/config.example.yaml
python scripts/02_bootstrap.py --config configs/config.example.yaml
python scripts/03_train_rf.py --config configs/config.example.yaml --strategy grid_search
python scripts/04_baselines.py --config configs/config.example.yaml
python scripts/05_feature_importance.py --config configs/config.example.yaml --mode full
```

