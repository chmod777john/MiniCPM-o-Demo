#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
PT_PATH="${PT_PATH:-/user/weihongliang/o5_weights/houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800.pt}"
PROMPT_WAV="${PROMPT_WAV:-/user/weihongliang/MiniCPM-o-4_6/assets/audio_cases/paimon__system_ref_audio.wav}"
VIDEO_PATH="${VIDEO_PATH:-/user/weihongliang/omni_demo_duplex_01.mp4}"
MAX_UNITS="${MAX_UNITS:-8}"
EVENT_TIMEOUT_S="${EVENT_TIMEOUT_S:-600}"
RUN_DIR="${RUN_DIR:-/user/weihongliang/o5_alignment_runs/api_vs_canonical_noaccel_video01_max${MAX_UNITS}_20260726_01}"
CANONICAL_DIR="${CANONICAL_DIR:-/user/weihongliang/thin_unified_runs/unified_trace_baseline_video01_max8_20260724_01}"

GATEWAY_PORT="${GATEWAY_PORT:-8061}"
GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8062}"
BACKEND_PORT="${BACKEND_PORT:-22661}"
WORKER_PORT="${WORKER_PORT:-22662}"
SERVICE_LOG_DIR="${SERVICE_LOG_DIR:-${RUN_DIR}/service_logs}"
COMPARE_STRICT="${COMPARE_STRICT:-0}"
COMPARE_PREFIX="${COMPARE_PREFIX:-1}"

if [ -z "${RUN_DIR}" ] || [ "${RUN_DIR}" = "/" ]; then
    echo "[api-align] invalid RUN_DIR=${RUN_DIR}" >&2
    exit 2
fi

rm -rf "${RUN_DIR}"
mkdir -p "${SERVICE_LOG_DIR}"
cd "${PROJECT_DIR}"

export PROJECT_DIR
export VENV_DIR
export MODEL_PATH
export PT_PATH
export ENABLE_FRP=0
export GPU_ID="${GPU_ID:-0}"
export GATEWAY_PORT
export GATEWAY_INTERNAL_PORT
export BACKEND_PORT
export WORKER_PORT
export LOG_DIR="${SERVICE_LOG_DIR}"
export O5_ATTN_IMPLEMENTATION="${O5_ATTN_IMPLEMENTATION:-sdpa}"
export O5_PRELOAD_BOTH_TTS="${O5_PRELOAD_BOTH_TTS:-0}"
export O5_STARTUP_SEED="${O5_STARTUP_SEED:-0}"
export O5_TOKEN_TRACE_PATH="${O5_TOKEN_TRACE_PATH:-${RUN_DIR}/api/token_trace.json}"
export O5_TTS_ARGMAX="${O5_TTS_ARGMAX:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TTS_N_TIMESTEPS="${TTS_N_TIMESTEPS:-5}"

PYTHON="${VENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "[api-align] missing python: ${PYTHON}" >&2
    exit 2
fi

echo "[api-align] project=${PROJECT_DIR}"
echo "[api-align] run=${RUN_DIR}"
echo "[api-align] canonical=${CANONICAL_DIR}"
echo "[api-align] model=${MODEL_PATH}"
echo "[api-align] pt=${PT_PATH}"
echo "[api-align] max_units=${MAX_UNITS}"
echo "[api-align] event_timeout_s=${EVENT_TIMEOUT_S}"
echo "[api-align] tts_n_timesteps=${TTS_N_TIMESTEPS}"
echo "[api-align] preload_both_tts=${O5_PRELOAD_BOTH_TTS}"
echo "[api-align] startup_seed=${O5_STARTUP_SEED}"
echo "[api-align] token_trace=${O5_TOKEN_TRACE_PATH}"
echo "[api-align] tts_argmax=${O5_TTS_ARGMAX}"

bash scripts/start_o5_cctl_service.sh > "${RUN_DIR}/start.log" 2>&1 &
service_pid=$!

cleanup() {
    kill "${service_pid}" 2>/dev/null || true
    wait "${service_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 900); do
    if curl -sk "https://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
        echo "[api-align] service ready"
        break
    fi
    if ! kill -0 "${service_pid}" 2>/dev/null; then
        echo "[api-align] service exited during startup" >&2
        tail -n 240 "${RUN_DIR}/start.log" >&2 || true
        exit 1
    fi
    sleep 2
done
curl -sk "https://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null

"${PYTHON}" -B tools/o5trace/api_canonical_video_probe.py \
    --url "https://127.0.0.1:${GATEWAY_PORT}" \
    --insecure \
    --video "${VIDEO_PATH}" \
    --prompt-wav "${PROMPT_WAV}" \
    --out-dir "${RUN_DIR}/api" \
    --max-units "${MAX_UNITS}" \
    --force-listen-count 0 \
    --decode-mode greedy \
    --event-timeout-s "${EVENT_TIMEOUT_S}" \
    --include-events

set +e
compare_args=()
if [ "${COMPARE_PREFIX}" = "1" ]; then
    compare_args+=(--compare-prefix)
fi
"${PYTHON}" -B tools/o5trace/compare_canonical_units.py \
    "${CANONICAL_DIR}" \
    "${RUN_DIR}/api" \
    --out-json "${RUN_DIR}/compare.json" \
    --out-md "${RUN_DIR}/compare.md" \
    "${compare_args[@]}"
compare_status=$?
set -e

cat "${RUN_DIR}/compare.json"
if [ "${COMPARE_STRICT}" = "1" ]; then
    exit "${compare_status}"
fi
exit 0
