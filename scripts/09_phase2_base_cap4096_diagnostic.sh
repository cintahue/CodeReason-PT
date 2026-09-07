#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/base_eval_v2.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_HOME="${HF_HOME:-/mnt/data/liangjunwei/CodeReason-PT/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/data/liangjunwei/CodeReason-PT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

IFS=',' read -r -a PHASE2_GPUS <<< "${PHASE2_GPUS:-0,1,2,3}"
SHARD_COUNT="${SHARD_COUNT:-${#PHASE2_GPUS[@]}}"
if [ "${#PHASE2_GPUS[@]}" -ne "${SHARD_COUNT}" ]; then
  echo "PHASE2_GPUS count must match SHARD_COUNT" >&2
  exit 1
fi

pids=()
for shard_index in "${!PHASE2_GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${PHASE2_GPUS[$shard_index]}" \
    "${PYTHON_BIN}" -m eval.diagnose_base_cap_extension \
      --config "${CONFIG}" \
      --shard-index "${shard_index}" \
      --shard-count "${SHARD_COUNT}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON_BIN}" -m eval.diagnose_base_cap_extension \
  --config "${CONFIG}" \
  --shard-count "${SHARD_COUNT}" \
  --combine-only
