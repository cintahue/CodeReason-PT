#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase0.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m data.build_raw --config "${CONFIG}"
"${PYTHON_BIN}" -m data.prepare --config "${CONFIG}"
"${PYTHON_BIN}" -m data.validate --config "${CONFIG}"
"${PYTHON_BIN}" -m data.audit --config "${CONFIG}" --output phase0_audit.json
