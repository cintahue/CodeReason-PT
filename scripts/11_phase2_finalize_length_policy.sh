#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/base_eval_v2.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m eval.finalize_phase2_length_policy --config "${CONFIG}"
