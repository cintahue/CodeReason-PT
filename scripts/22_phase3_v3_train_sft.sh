#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/sft_v3.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/mnt/data/liangjunwei/CodeReason-PT/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/data/liangjunwei/CodeReason-PT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

"${PYTHON_BIN}" -m sft.train_v3 --config "${CONFIG}" --mode train
"${PYTHON_BIN}" -m sft.verify_v3_checkpoint --config "${CONFIG}"
