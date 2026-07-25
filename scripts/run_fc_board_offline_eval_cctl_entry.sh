#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel/bin/python}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
PT_PATH="${PT_PATH:-/user/weihongliang/o5_weights/job616069_iter0000500/iter_0000500_o5.pt}"
DATA_DIR="${DATA_DIR:-/user/weihongliang/o5_fc_assets/board_mvp_20260724/delivery_train_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/user/weihongliang/fc_board_offline_runs}"
LIMIT="${LIMIT:-3}"
GPU_NUM="${GPU_NUM:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
DECODE_MODE="${DECODE_MODE:-greedy}"
SKIP_MUTATED="${SKIP_MUTATED:-1}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/20260725_iter500_limit${LIMIT}_${STAMP}}"
export OUTPUT_DIR

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

args=(
  evaluate_fc_duplex_batch.py
  --model-path "${MODEL_PATH}"
  --pt-path "${PT_PATH}"
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --decode-mode "${DECODE_MODE}"
  --limit "${LIMIT}"
  --gpu-num "${GPU_NUM}"
)

if [[ "${SKIP_MUTATED}" == "1" || "${SKIP_MUTATED}" == "true" ]]; then
  args+=(--skip-mutated)
fi

echo "[fc-board-eval] project=${PROJECT_DIR}"
echo "[fc-board-eval] python=${PYTHON}"
echo "[fc-board-eval] model=${MODEL_PATH}"
echo "[fc-board-eval] pt=${PT_PATH}"
echo "[fc-board-eval] data=${DATA_DIR}"
echo "[fc-board-eval] output=${OUTPUT_DIR}"
echo "[fc-board-eval] limit=${LIMIT} gpu_num=${GPU_NUM} skip_mutated=${SKIP_MUTATED}"

"${PYTHON}" "${args[@]}"

echo "EVAL_OUT=${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  "${PYTHON}" - <<'PY'
import json
import os

path = os.path.join(os.environ["OUTPUT_DIR"], "summary.json")
data = json.load(open(path, encoding="utf-8"))
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
fi
