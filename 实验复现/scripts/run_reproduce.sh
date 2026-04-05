#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-demo}"
CONFIG="${2:-configs/config.demo.yaml}"
PYTHON_EXE="${PYTHON_EXE:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"
echo "[INFO] RepoRoot: $REPO_ROOT"
echo "[INFO] Python:   $PYTHON_EXE"
echo "[INFO] Mode:     $MODE"
echo "[INFO] Config:   $CONFIG"

"$PYTHON_EXE" ./scripts/run_reproduce.py --mode "$MODE" --config "$CONFIG"
echo "[DONE] Pipeline mode '$MODE' finished."

