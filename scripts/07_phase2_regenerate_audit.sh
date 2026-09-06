#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/base_eval.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m eval.regenerate_phase2_audit --config "${CONFIG}"
