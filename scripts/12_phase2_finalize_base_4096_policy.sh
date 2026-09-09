#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/base_eval_v2.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m eval.finalize_base_4096_policy --config "${CONFIG}"
