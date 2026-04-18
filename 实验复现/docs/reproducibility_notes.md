# Reproducibility Notes

## Scope
This package focuses on computational reproducibility of the manuscript methods.
It does not include raw source datasets and does not provide publication/submission artifacts.

## Expected Variability
- Public database updates may change available samples and gene coverage.
- Minor differences in preprocessing-compatible software versions can affect exact metrics.
- Variance-filtered expression feature counts should be near the manuscript scale when data versions and filtering logic match.

## Runtime Behavior Without Data
Scripts are executable but require user-supplied public data files in `data/input/`.
If files are missing or malformed, scripts fail with explicit validation messages.

## Enrichment Dependencies
`script 10` requires R packages:
- clusterProfiler
- org.Hs.eg.db
- yaml

Gene-symbol to Entrez conversion is performed before KEGG/GO enrichment.
If mapping coverage is poor, enrichment may return empty tables.
