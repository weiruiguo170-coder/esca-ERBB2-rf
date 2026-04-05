#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[INFO] Build submission figures/tables..."
"$PYTHON_EXE" ./src/project_cli.py --mode submission --python "$PYTHON_EXE"
"$PYTHON_EXE" ./src/project_cli.py --mode figure4_sensitivity --python "$PYTHON_EXE"
echo "[DONE] Submission assets generated."
