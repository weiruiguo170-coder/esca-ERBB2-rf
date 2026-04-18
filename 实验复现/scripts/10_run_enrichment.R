args <- commandArgs(trailingOnly = TRUE)
config_path <- if (length(args) >= 1) args[[1]] else "configs/config.yaml"

if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("Package 'yaml' is required. Install it before running enrichment.")
}

cfg <- yaml::read_yaml(config_path)
paths <- cfg$paths

required_pkgs <- c("clusterProfiler", "org.Hs.eg.db")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(
    paste0(
      "Missing required R package(s): ",
      paste(missing_pkgs, collapse = ", "),
      ". Install dependencies listed in README before running script 10."
    )
  )
}

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
})

read_gene_file <- function(path) {
  genes <- trimws(readLines(path, warn = FALSE))
  genes <- genes[genes != ""]
  unique(genes)
}

write_empty_result <- function(path) {
  empty <- data.frame(
    ID = character(),
    Description = character(),
    pvalue = numeric(),
    p.adjust = numeric(),
    geneID = character(),
    Count = integer()
  )
  write.csv(empty, path, row.names = FALSE)
}

project_root <- normalizePath(file.path(dirname(config_path), ".."), winslash = "/", mustWork = FALSE)
resolve_path <- function(rel_path) normalizePath(file.path(project_root, rel_path), winslash = "/", mustWork = FALSE)

input_top <- resolve_path(paths$top10_genes)
input_bg <- resolve_path(paths$background_genes)
out_dir <- resolve_path(paths$enrichment_dir)
out_kegg <- resolve_path(paths$kegg_results)
out_go <- resolve_path(paths$go_results)
out_mapping <- resolve_path(paths$enrichment_mapping)

if (!file.exists(input_top) || !file.exists(input_bg)) {
  stop("Top gene file or background gene file is missing. Run script 09 first.")
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

top_genes <- read_gene_file(input_top)
background_genes <- read_gene_file(input_bg)

if (length(top_genes) == 0 || length(background_genes) == 0) {
  stop("Top genes or background genes are empty.")
}

top_map <- clusterProfiler::bitr(
  top_genes,
  fromType = "SYMBOL",
  toType = "ENTREZID",
  OrgDb = org.Hs.eg.db
)
bg_map <- clusterProfiler::bitr(
  background_genes,
  fromType = "SYMBOL",
  toType = "ENTREZID",
  OrgDb = org.Hs.eg.db
)

if (nrow(top_map) == 0 || nrow(bg_map) == 0) {
  stop("Gene ID conversion produced zero mappings. Check gene symbols and species.")
}

p_cutoff <- as.numeric(cfg$enrichment$p_cutoff)
fdr_cutoff <- as.numeric(cfg$enrichment$fdr_cutoff)
go_ont <- cfg$enrichment$go_ontology

kegg_result <- tryCatch(
  {
    clusterProfiler::enrichKEGG(
      gene = unique(top_map$ENTREZID),
      organism = "hsa",
      universe = unique(bg_map$ENTREZID),
      pvalueCutoff = p_cutoff,
      qvalueCutoff = fdr_cutoff
    )
  },
  error = function(e) NULL
)

go_result <- tryCatch(
  {
    clusterProfiler::enrichGO(
      gene = unique(top_map$ENTREZID),
      universe = unique(bg_map$ENTREZID),
      OrgDb = org.Hs.eg.db,
      keyType = "ENTREZID",
      ont = go_ont,
      pvalueCutoff = p_cutoff,
      qvalueCutoff = fdr_cutoff,
      readable = TRUE
    )
  },
  error = function(e) NULL
)

export_result <- function(enrich_obj, output_path) {
  if (is.null(enrich_obj) || nrow(as.data.frame(enrich_obj)) == 0) {
    write_empty_result(output_path)
    return(invisible(NULL))
  }

  result_df <- as.data.frame(enrich_obj)
  result_df <- result_df[result_df$pvalue < p_cutoff & result_df$p.adjust < fdr_cutoff, ]

  if (nrow(result_df) == 0) {
    write_empty_result(output_path)
  } else {
    write.csv(result_df, output_path, row.names = FALSE)
  }
}

export_result(kegg_result, out_kegg)
export_result(go_result, out_go)

mapping_summary <- data.frame(
  input_top_gene_count = length(top_genes),
  mapped_top_gene_count = length(unique(top_map$SYMBOL)),
  input_background_gene_count = length(background_genes),
  mapped_background_gene_count = length(unique(bg_map$SYMBOL)),
  stringsAsFactors = FALSE
)
write.csv(mapping_summary, out_mapping, row.names = FALSE)

cat("[10] Enrichment completed.\n")
cat(paste0("[10] KEGG file: ", out_kegg, "\n"))
cat(paste0("[10] GO file: ", out_go, "\n"))
