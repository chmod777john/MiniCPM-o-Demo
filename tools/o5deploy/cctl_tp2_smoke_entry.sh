#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
PYTHON="${VENV_DIR}/bin/python"
export VENV_DIR

export MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
export PT_PATH="${PT_PATH:-/user/weihongliang/o5_weights/houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800.pt}"
export BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_weights/o5_backbone_hf_houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800}"
export O5_BACKBONE_DIR="${BACKBONE_DIR}"
export O5_DEPLOY_MODE="${O5_DEPLOY_MODE:-tp2}"
export O5_ATTN_IMPLEMENTATION="${O5_ATTN_IMPLEMENTATION:-auto}"
export O5_LLM_CACHE="${O5_LLM_CACHE:-32768}"
export O5_LLM_GRAPH="${O5_LLM_GRAPH:-1}"
export O5_SPMD_HEARTBEAT_INTERVAL="${O5_SPMD_HEARTBEAT_INTERVAL:-30}"
export ENABLE_FRP=0

export GATEWAY_PORT="${GATEWAY_PORT:-18091}"
export GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-18092}"
export BACKEND_PORT="${BACKEND_PORT:-18093}"
export WORKER_PORT="${WORKER_PORT:-18094}"
export WORKER_ID="${WORKER_ID:-o5-clean-llmgraph-smoke-worker}"
export WORKER_GPU_GROUP="${WORKER_GPU_GROUP:-cctl-a100-tp2-smoke}"

VIDEO_PATH="${VIDEO_PATH:-/user/weihongliang/omni_demo_duplex_01.mp4}"
AUDIO_WAV_PATH="${AUDIO_WAV_PATH:-${PROJECT_DIR}/examples/realtime/assets/test.wav}"
MAX_SESSION_S="${MAX_SESSION_S:-90}"
PROBE_MODE="${PROBE_MODE:-video}"
VIDEO_REPEAT="${VIDEO_REPEAT:-1}"
AUDIO_REPEAT="${AUDIO_REPEAT:-1}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/run-logs/cctl_tp2_smoke_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"
export LOG_DIR="${OUT_DIR}/service"

echo "[smoke] project=${PROJECT_DIR}"
echo "[smoke] model=${MODEL_PATH}"
echo "[smoke] pt=${PT_PATH}"
echo "[smoke] backbone=${BACKBONE_DIR}"
echo "[smoke] video=${VIDEO_PATH}"
echo "[smoke] audio=${AUDIO_WAV_PATH}"
echo "[smoke] probe_mode=${PROBE_MODE} video_repeat=${VIDEO_REPEAT} audio_repeat=${AUDIO_REPEAT}"
echo "[smoke] out=${OUT_DIR}"

if [ ! -x "${PYTHON}" ]; then echo "[smoke] missing python: ${PYTHON}" >&2; exit 1; fi
if [ ! -f "${VIDEO_PATH}" ]; then echo "[smoke] missing video: ${VIDEO_PATH}" >&2; exit 1; fi
if [ ! -f "${AUDIO_WAV_PATH}" ]; then echo "[smoke] missing audio: ${AUDIO_WAV_PATH}" >&2; exit 1; fi

bash scripts/start_o5_tp2_cctl_service.sh > "${OUT_DIR}/service.log" 2>&1 &
service_pid=$!

cleanup() {
    if kill -0 "${service_pid}" 2>/dev/null; then
        kill "${service_pid}" 2>/dev/null || true
        wait "${service_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 900); do
    if grep -q "service ready" "${OUT_DIR}/service.log" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "${service_pid}" 2>/dev/null; then
        echo "[smoke] service exited before ready" >&2
        tail -n 200 "${OUT_DIR}/service.log" >&2 || true
        exit 1
    fi
    sleep 2
done

if [ "${ready}" != "1" ]; then
    echo "[smoke] timeout waiting for service ready" >&2
    tail -n 200 "${OUT_DIR}/service.log" >&2 || true
    exit 1
fi

run_video_probe() {
    local idx="$1"
    local result="${OUT_DIR}/video_probe_result_${idx}.json"
    echo "[smoke] running realtime video probe #${idx}"
    "${PYTHON}" examples/realtime/video_probe.py \
        --url "https://127.0.0.1:${GATEWAY_PORT}" \
        --video "${VIDEO_PATH}" \
        --insecure \
        --max-session-s "${MAX_SESSION_S}" \
        --stop-on-end-of-turn \
        --pretty-json \
        > "${result}"
    cat "${result}"
}

run_audio_probe() {
    local idx="$1"
    local result="${OUT_DIR}/audio_probe_result_${idx}.json"
    echo "[smoke] running realtime audio probe #${idx}"
    "${PYTHON}" examples/realtime/audio_probe.py \
        --url "https://127.0.0.1:${GATEWAY_PORT}" \
        --input-wav "${AUDIO_WAV_PATH}" \
        --insecure \
        --max-session-s "${MAX_SESSION_S}" \
        --pretty-json \
        > "${result}"
    cat "${result}"
}

echo "[smoke] service ready; running realtime probes"
case "${PROBE_MODE}" in
    video)
        for idx in $(seq 1 "${VIDEO_REPEAT}"); do run_video_probe "${idx}"; done
        ;;
    audio)
        for idx in $(seq 1 "${AUDIO_REPEAT}"); do run_audio_probe "${idx}"; done
        ;;
    both)
        for idx in $(seq 1 "${VIDEO_REPEAT}"); do run_video_probe "${idx}"; done
        for idx in $(seq 1 "${AUDIO_REPEAT}"); do run_audio_probe "${idx}"; done
        ;;
    *)
        echo "[smoke] invalid PROBE_MODE=${PROBE_MODE}; expected video, audio, or both" >&2
        exit 1
        ;;
esac
echo "[smoke] done"
