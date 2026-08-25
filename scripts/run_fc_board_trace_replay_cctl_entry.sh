#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
PT_PATH="${PT_PATH:-/user/weihongliang/o5_weights/job616069_iter0000500/iter_0000500_o5.pt}"
BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_backbones/job616069_iter0000500_hf}"
CASE_PATH="${CASE_PATH:-/user/weihongliang/o5_fc_assets/board_mvp_20260724/delivery_train_data/dob_midtrain_v1_20260628_animal_seed_ct01_and_04702.json}"
MODE="${MODE:-single_eager}"
RUN_ID="${RUN_ID:-fc-board-trace-${MODE}-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/user/weihongliang/fc_align_enhance_fc_runs/${RUN_ID}}"
TRACE_DIR="${TRACE_DIR:-${OUT_DIR}/trace_sessions}"
REFERENCE_SESSION="${REFERENCE_SESSION:-}"

if [[ "${MODE}" != "single_eager" && "${MODE}" != "tp2" ]]; then
  echo "unsupported MODE=${MODE}; expected single_eager or tp2" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "${TRACE_DIR}" "${OUT_DIR}/service_logs"
cd "${PROJECT_DIR}"
export PROJECT_DIR VENV_DIR MODEL_PATH PT_PATH BACKBONE_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export O5_DEPLOY_MODE="${MODE}"
export O5_BACKBONE_DIR="${BACKBONE_DIR}"
export O5_LLM_CACHE="${O5_LLM_CACHE:-32768}"
export O5_EXPERTS_IMPLEMENTATION="${O5_EXPERTS_IMPLEMENTATION:-eager}"
export O5_LLM_GRAPH="${O5_LLM_GRAPH:-0}"
export O5_TTS_GRAPH="${O5_TTS_GRAPH:-0}"
export O5_VOCODER_GRAPH="${O5_VOCODER_GRAPH:-0}"
export O5_TTS_FAST="${O5_TTS_FAST:-0}"
export O5_LMHEAD="${O5_LMHEAD:-0}"
export O5_FUSE_VISION_AUDIO="${O5_FUSE_VISION_AUDIO:-0}"
export O5_VISION_BATCH="${O5_VISION_BATCH:-0}"
export O5_DETERMINISTIC_REPLAY="${O5_DETERMINISTIC_REPLAY:-1}"
export O5_SESSION_SEED="${O5_SESSION_SEED:-0}"
export O5_TTS_ARGMAX="${O5_TTS_ARGMAX:-1}"
export O5_SESSION_TRACE_MODE=replay
export O5_TOKEN_TRACE_DIR="${TRACE_DIR}"
export O5_REPLAY_FORCING="${O5_REPLAY_FORCING:-none}"
if [[ -n "${REFERENCE_SESSION}" ]]; then
  export O5_REPLAY_REFERENCE="${REFERENCE_SESSION}"
fi

export GATEWAY_PORT="${GATEWAY_PORT:-8024}"
export GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8025}"
export BACKEND_PORT="${BACKEND_PORT:-22514}"
export WORKER_PORT="${WORKER_PORT:-22414}"
export WORKER_ID="${WORKER_ID:-o5-fc-trace-${MODE}-worker}"
export WORKER_GPU_GROUP="${WORKER_GPU_GROUP:-cctl-a100-fc-trace-${MODE}}"
export LOG_DIR="${OUT_DIR}/service_logs"
export ENABLE_FRP=0

echo "[fc-trace] repo=${PROJECT_DIR}"
echo "[fc-trace] commit=$(git rev-parse HEAD) branch=$(git branch --show-current)"
echo "[fc-trace] python=${VENV_DIR}/bin/python mode=${MODE}"
echo "[fc-trace] model=${MODEL_PATH} pt=${PT_PATH} backbone=${BACKBONE_DIR}"
echo "[fc-trace] case=${CASE_PATH} out=${OUT_DIR} trace=${TRACE_DIR}"
echo "[fc-trace] flags experts=${O5_EXPERTS_IMPLEMENTATION} llm_graph=${O5_LLM_GRAPH} tts_graph=${O5_TTS_GRAPH} vocoder_graph=${O5_VOCODER_GRAPH} tts_fast=${O5_TTS_FAST} lmhead=${O5_LMHEAD} vision_batch=${O5_VISION_BATCH} forcing=${O5_REPLAY_FORCING}"

if [[ "${MODE}" == "tp2" ]]; then
  bash scripts/start_o5_tp2_cctl_service.sh >"${OUT_DIR}/service.log" 2>&1 &
else
  export GPU_ID="${GPU_ID:-0}"
  bash scripts/start_o5_cctl_service.sh >"${OUT_DIR}/service.log" 2>&1 &
fi
service_pid=$!
cleanup() {
  kill "${service_pid}" 2>/dev/null || true
  wait "${service_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + 1800))
until curl -sf "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${service_pid}" 2>/dev/null; then
    echo "[fc-trace] service exited before ready" >&2
    tail -n 240 "${OUT_DIR}/service.log" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "[fc-trace] timeout waiting for backend" >&2
    tail -n 240 "${OUT_DIR}/service.log" >&2 || true
    exit 1
  fi
  sleep 5
done

"${VENV_DIR}/bin/python" scripts/probe_fc_board_backend.py \
  --backend "http://127.0.0.1:${BACKEND_PORT}" \
  --path /backend \
  --case "${CASE_PATH}" \
  --out "${OUT_DIR}/probe.json" \
  --normalize-tools \
  --tool-response-schedule gt \
  --final-max-wait "${FINAL_MAX_WAIT:-300}" \
  --use-case-ref-audio \
  --generate-audio

latest_session="$(find "${TRACE_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'sess_*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -n "${latest_session}" ]]; then
  printf '%s\n' "${latest_session}" > "${OUT_DIR}/trace_session_path.txt"
  echo "[fc-trace] trace_session=${latest_session}"
else
  echo "[fc-trace] no trace session directory found" >&2
  exit 1
fi
echo "[fc-trace] result=${OUT_DIR}/probe.json"
