#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-high-cu128-vllm021}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
PT_PATH="${PT_PATH:-/user/weihongliang/o5_weights/chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_sft2_8k_audio_online_process_on_online_audio_process_v2_iter_100.pt}"
BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_weights/o5_backbone_hf_chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_sft2_8k_audio_online_process_on_online_audio_process_v2_iter_100}"
VIDEO_PATH="${VIDEO_PATH:-/user/weihongliang/omni_demo_duplex_01.mp4}"
PROMPT_WAV="${PROMPT_WAV:-/user/weihongliang/MiniCPM-o-4_6/assets/audio_cases/paimon__system_ref_audio.wav}"
MAX_UNITS="${MAX_UNITS:-8}"
RUN_DIR="${RUN_DIR:-/user/weihongliang/o5_strategy_hd_grouped_api_20260822}"

GATEWAY_PORT="${GATEWAY_PORT:-8009}"
GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8010}"
BACKEND_PORT="${BACKEND_PORT:-22510}"
WORKER_PORT="${WORKER_PORT:-22410}"
START_TIMEOUT_S="${START_TIMEOUT_S:-1800}"
EVENT_TIMEOUT_S="${EVENT_TIMEOUT_S:-900}"

PYTHON="${VENV_DIR}/bin/python"
for path_name in MODEL_PATH PT_PATH BACKBONE_DIR VIDEO_PATH PROMPT_WAV; do
    path_value="${!path_name}"
    if [ ! -e "${path_value}" ]; then
        echo "[strategy-hd-probe] missing ${path_name}=${path_value}" >&2
        exit 2
    fi
done
if [ ! -x "${PYTHON}" ]; then
    echo "[strategy-hd-probe] missing python=${PYTHON}" >&2
    exit 2
fi

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}/service_logs"
cd "${PROJECT_DIR}"

export PROJECT_DIR VENV_DIR MODEL_PATH PT_PATH BACKBONE_DIR
export GATEWAY_PORT GATEWAY_INTERNAL_PORT BACKEND_PORT WORKER_PORT
export ENABLE_FRP=0
export LOG_DIR="${RUN_DIR}/service_logs"
export O5_DEPLOY_MODE=tp2
export O5_LLM_CACHE=32768
export O5_EXPERTS_IMPLEMENTATION=hybrid
export O5_GROUPED_PREFILL_MIN_TOKENS=100
export O5_ATTN_IMPLEMENTATION=sdpa
export O5_LLM_GRAPH=1
export O5_TTS_GRAPH=1
export O5_VOCODER_GRAPH=0
export O5_TTS_FAST=1
export O5_LMHEAD=1
export O5_FUSE_VISION_AUDIO=1
export O5_VISION_BATCH=1
export O5_SPMD_HEARTBEAT_INTERVAL=30
export O5_TOKEN_TRACE_DIR="${RUN_DIR}/sessions"
export O5_SESSION_TRACE_MODE=tokens
export O5_MOE_HYBRID_TRACE=1
export TOKENIZERS_PARALLELISM=false

COMMIT="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current)"
{
    echo "project=${PROJECT_DIR}"
    echo "branch=${BRANCH}"
    echo "commit=${COMMIT}"
    echo "venv=${VENV_DIR}"
    echo "model=${MODEL_PATH}"
    echo "pt=${PT_PATH}"
    echo "backbone=${BACKBONE_DIR}"
    echo "video=${VIDEO_PATH}"
    echo "prompt_wav=${PROMPT_WAV}"
    echo "experts=${O5_EXPERTS_IMPLEMENTATION} grouped_prefill_min_tokens=${O5_GROUPED_PREFILL_MIN_TOKENS}"
    echo "strategy_hd=true strategy_hd_max_slice_nums=4"
    echo "llm_graph=${O5_LLM_GRAPH} tts_graph=${O5_TTS_GRAPH} vocoder_graph=${O5_VOCODER_GRAPH}"
    echo "llm_cache=${O5_LLM_CACHE} attn=${O5_ATTN_IMPLEMENTATION}"
} | tee "${RUN_DIR}/run_config.txt"

bash "${PROJECT_DIR}/scripts/start_o5_tp2_cctl_service.sh" >"${RUN_DIR}/service.log" 2>&1 &
service_pid=$!
cleanup() {
    kill "${service_pid}" 2>/dev/null || true
    wait "${service_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + START_TIMEOUT_S))
until curl -skf "https://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${service_pid}" 2>/dev/null; then
        echo "[strategy-hd-probe] service exited during startup" >&2
        tail -n 300 "${RUN_DIR}/service.log" >&2 || true
        exit 1
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "[strategy-hd-probe] service startup timed out" >&2
        tail -n 300 "${RUN_DIR}/service.log" >&2 || true
        exit 1
    fi
    sleep 2
done
echo "[strategy-hd-probe] service ready"

"${PYTHON}" -B tools/o5trace/api_canonical_video_probe.py \
    --url "https://127.0.0.1:${GATEWAY_PORT}" \
    --insecure \
    --video "${VIDEO_PATH}" \
    --prompt-wav "${PROMPT_WAV}" \
    --out-dir "${RUN_DIR}/api" \
    --max-units "${MAX_UNITS}" \
    --strategy-hd \
    --strategy-hd-max-slice-nums 4 \
    --decode-mode greedy \
    --temperature 0.0 \
    --top-k 0 \
    --top-p 1.0 \
    --force-listen-count 0 \
    --event-timeout-s "${EVENT_TIMEOUT_S}" \
    --include-events \
    | tee "${RUN_DIR}/probe.log"

"${PYTHON}" - "${RUN_DIR}/api/events.json" "${RUN_DIR}/unit_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

events = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = []
for event in events:
    if event.get("type") != "response.output.delta":
        continue
    metrics = event.get("metrics") or {}
    if not metrics or not event.get("input_id"):
        continue
    rows.append({
        "input_id": event["input_id"],
        "kind": event.get("kind"),
        "effective_max_slice_nums": metrics.get("effective_max_slice_nums"),
        "strategy_hd_next_max_slice_nums": metrics.get("strategy_hd_next_max_slice_nums"),
        "prefill_ms": metrics.get("prefill_ms"),
        "wall_clock_ms": metrics.get("wall_clock_ms"),
        "cost_llm_ms": metrics.get("cost_llm_ms"),
        "cost_tts_ms": metrics.get("cost_tts_ms"),
        "cost_token2wav_ms": metrics.get("cost_token2wav_ms"),
        "n_tokens": metrics.get("n_tokens"),
        "n_tts_tokens": metrics.get("n_tts_tokens"),
    })
dedup = {}
for row in rows:
    dedup[(row["input_id"], row["kind"])] = row
output = list(dedup.values())
Path(sys.argv[2]).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"metric_events": len(rows), "unique_unit_kind_rows": len(output), "rows": output}, ensure_ascii=False, indent=2))
PY

echo "[strategy-hd-probe] completed run_dir=${RUN_DIR}"
