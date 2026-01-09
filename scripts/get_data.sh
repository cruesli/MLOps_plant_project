#!/usr/bin/env bash
set -euo pipefail

DATASET="mohitsingh1804/plantvillage"
OUTDIR="data/raw"
ZIP="${OUTDIR}/plantvillage.zip"

mkdir -p "${OUTDIR}"

# Download (Kaggle CLI provided by uv)
uv run kaggle datasets download -d "${DATASET}" -p "${OUTDIR}"

# Extract
if command -v unzip >/dev/null 2>&1; then
  unzip -q "${ZIP}" -d "${OUTDIR}"
else
  uv run python -m zipfile -e "${ZIP}" "${OUTDIR}"
fi

echo "Done. Data is in ${OUTDIR}"
