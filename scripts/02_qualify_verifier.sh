#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase0.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-16}"

"${PYTHON_BIN}" -m verifier.qualify_phase1 --config "${CONFIG}" --workers "${WORKERS}"
