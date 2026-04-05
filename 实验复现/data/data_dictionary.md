# Input Data Dictionary

- `CELL_LINE_ID`: cell line identifier (string)
- `TISSUE`: tissue/source label, used for esophagus/upper-digestive filtering
- `DRUG_NAME`: drug name (should map to `Trastuzumab` and `Lapatinib`)
- `IC50`: raw IC50 value (positive numeric); pipeline converts to `log2(IC50)`
- `EXP_ERBB2`: expression feature for ERBB2
- `CNV_ERBB2`: copy-number feature for ERBB2
- `EXP_GRB7`: expression feature for GRB7
- `EXP_ERBB3`: expression feature for ERBB3
- `EXP_PIK3CA`: expression feature for PIK3CA
- `EXP_AKT1`: expression feature for AKT1
- `EXP_MAPK1`: expression feature for MAPK1
- `EXP_PTK6`: expression feature for PTK6
- `EXP_CCND1`: expression feature for CCND1
- `EXP_SHC1`: expression feature for SHC1

Only a template is provided in this repository. Real data must be prepared by users.

