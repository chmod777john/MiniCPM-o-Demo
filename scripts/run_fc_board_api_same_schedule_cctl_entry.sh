#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

RUN_ID="${RUN_ID:-board_api_same_schedule_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/user/weihongliang/fc_board_api_runs/${RUN_ID}}"
CASE_PATH="${CASE_PATH:-/user/weihongliang/o5_fc_assets/board_mvp_20260724/delivery_train_data/dob_midtrain_v1_20260628_animal_seed_ct01_and_04702.json}"

export VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
export MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
export PT_PATH="${PT_PATH:-/user/weihongliang/o5_weights/job616069_iter0000500/iter_0000500_o5.pt}"
export BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_backbones/job616069_iter0000500_hf}"

export O5_DEPLOY_MODE="${O5_DEPLOY_MODE:-tp2}"
export O5_ATTN_IMPLEMENTATION="${O5_ATTN_IMPLEMENTATION:-auto}"
export O5_LLM_CACHE="${O5_LLM_CACHE:-32768}"
export O5_LLM_GRAPH="${O5_LLM_GRAPH:-1}"
export O5_SPMD_HEARTBEAT_INTERVAL="${O5_SPMD_HEARTBEAT_INTERVAL:-30}"

export GATEWAY_PORT="${GATEWAY_PORT:-8024}"
export GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8025}"
export BACKEND_PORT="${BACKEND_PORT:-22514}"
export WORKER_PORT="${WORKER_PORT:-22414}"
export WORKER_ID="${WORKER_ID:-o5-fc-board-api-probe-worker}"
export WORKER_GPU_GROUP="${WORKER_GPU_GROUP:-cctl-a100-tp2-fc-board-api-probe}"
export LOG_DIR="${LOG_DIR:-${OUT_DIR}/service_logs}"
export ENABLE_FRP=0

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
echo "[same-schedule] out=${OUT_DIR}"
echo "[same-schedule] case=${CASE_PATH}"
echo "[same-schedule] pt=${PT_PATH}"

bash run_o5_tp2_fc_board_job616069_entry.sh &
service_pid=$!
cleanup() {
    kill "${service_pid}" 2>/dev/null || true
    wait "${service_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + 1800))
until curl -sf "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${service_pid}" 2>/dev/null; then
        echo "[same-schedule] service exited before ready" >&2
        tail -n 200 "${LOG_DIR}/backend_tp2.log" >&2 || true
        exit 1
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "[same-schedule] timeout waiting backend" >&2
        tail -n 200 "${LOG_DIR}/backend_tp2.log" >&2 || true
        exit 1
    fi
    sleep 5
done
echo "[same-schedule] backend ready"

"${VENV_DIR}/bin/python" scripts/probe_fc_board_backend.py \
    --backend "http://127.0.0.1:${BACKEND_PORT}" \
    --path /backend \
    --case "${CASE_PATH}" \
    --out "${OUT_DIR}/board_same_schedule.json" \
    --normalize-tools \
    --tool-response-schedule "${TOOL_RESPONSE_SCHEDULE:-gt}" \
    --final-max-wait "${FINAL_MAX_WAIT:-240}" \
    ${GENERATE_AUDIO:+--generate-audio} \
    ${USE_CASE_REF_AUDIO:+--use-case-ref-audio}

echo "[same-schedule] done ${OUT_DIR}/board_same_schedule.json"
