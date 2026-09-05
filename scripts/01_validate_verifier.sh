#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase0.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m verifier.validate_reference_solutions --config "${CONFIG}"
