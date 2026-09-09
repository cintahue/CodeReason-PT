#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/sft.yaml}"
MODE="${2:-dev}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# The frozen Phase 2 protocol evaluated the 3B model on one visible GPU.
# Keeping the same placement avoids device_map=auto sharding and inter-GPU
# transfers for a model that fits comfortably on a single RTX 3090.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/mnt/data/liangjunwei/CodeReason-PT/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/data/liangjunwei/CodeReason-PT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

printf 'SFT evaluation mode=%s CUDA_VISIBLE_DEVICES=%s\n' "${MODE}" "${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" -m eval.run_sft_dev --config "${CONFIG}" --mode "${MODE}"
