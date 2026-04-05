# Data Description (Strict Publish Policy)

## Repository policy

This repository does **not** include:

- real raw data from GDSC/CCLE
- any suspected real tabular/binary data artifacts
- intermediate outputs, model artifacts, or training outputs

## What is included

- `data/example_input_template.csv` (header-only template)
- `data/data_dictionary.md` (column definitions)
- docs, scripts, configs, and source code

## How users should prepare data

1. Download data from GDSC/CCLE on their own.
2. Prepare a merged input file that follows the template columns.
3. Put local input file into `data/input/` and run the pipeline.

## Demo behavior

Demo scripts may generate temporary local CSV files under `data/local/`.
These generated files are local-only and excluded from version control.

