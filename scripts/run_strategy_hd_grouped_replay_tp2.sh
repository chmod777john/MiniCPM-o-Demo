#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-high-cu128}"
SESSION_DIR="${SESSION_DIR:-/user/weihongliang/o5_replay_strategy_hd_input_8u_20260822}"
OUT_DIR="${OUT_DIR:-/user/weihongliang/o5_replay_strategy_hd_runs_20260822/demo-tp2-free}"
REFERENCE_SESSION="${REFERENCE_SESSION:-}"
FORCING="${FORCING:-none}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/user/weihongliang/o5_weights/o5_full_hf_chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_sft2_8k_audio_online_process_on_online_audio_process_v2_iter_100}"
ASSETS_DIR="${ASSETS_DIR:-/user/weihongliang/MiniCPM-o-4_6/assets}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/user/weihongliang/MiniCPM-o-4_6}"
TOKEN2WAV_DIR="${TOKEN2WAV_DIR:-/user/weihongliang/o5_model_assets/token2wav}"

PYTHON="${VENV_DIR}/bin/python"
TORCHRUN="${VENV_DIR}/bin/torchrun"
for path_name in PYTHON TORCHRUN SESSION_DIR WEIGHTS_DIR ASSETS_DIR CANONICAL_ROOT TOKEN2WAV_DIR; do
    path_value="${!path_name}"
    if [ ! -e "${path_value}" ]; then
        echo "[strategy-hd-replay-tp2] missing ${path_name}=${path_value}" >&2
        exit 2
    fi
done

export PROJECT_DIR VENV_DIR O5_DEPLOY_MODE=tp2 O5_EXPERTS_IMPLEMENTATION=hybrid
export O5_GROUPED_PREFILL_MIN_TOKENS=100 O5_LLM_CACHE=32768
export O5_LLM_GRAPH=1 O5_TTS_GRAPH=1 O5_VOCODER_GRAPH=0 O5_TTS_FAST=1
export O5_LMHEAD=1 O5_FUSE_VISION_AUDIO=1 O5_VISION_BATCH=1
export O5_UNIT_PREFILL_BATCH="${O5_UNIT_PREFILL_BATCH:-1}"
export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

echo "[strategy-hd-replay-tp2] project=${PROJECT_DIR}"
echo "[strategy-hd-replay-tp2] commit=$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
echo "[strategy-hd-replay-tp2] branch=$(git -C "${PROJECT_DIR}" branch --show-current)"
echo "[strategy-hd-replay-tp2] venv=${VENV_DIR}"
echo "[strategy-hd-replay-tp2] session=${SESSION_DIR} out=${OUT_DIR} reference=${REFERENCE_SESSION:-none} forcing=${FORCING}"
echo "[strategy-hd-replay-tp2] weights=${WEIGHTS_DIR} assets=${ASSETS_DIR}"
echo "[strategy-hd-replay-tp2] experts=${O5_EXPERTS_IMPLEMENTATION} threshold=${O5_GROUPED_PREFILL_MIN_TOKENS} unit_prefill_batch=${O5_UNIT_PREFILL_BATCH} strategy_hd=1 max_slice=4"

cd "${PROJECT_DIR}"
REPLAY_ARGS=(
    --session-dir "${SESSION_DIR}"
    --out-dir "${OUT_DIR}"
    --target demo-tp2
    --forcing "${FORCING}"
    --capture-mode replay
    --max-units 8
    --overwrite
    --canonical-root "${CANONICAL_ROOT}"
    --token2wav-dir "${TOKEN2WAV_DIR}"
    --weights-dir "${WEIGHTS_DIR}"
    --assets-dir "${ASSETS_DIR}"
)
if [ -n "${REFERENCE_SESSION}" ]; then
    REPLAY_ARGS+=(--reference-session "${REFERENCE_SESSION}")
fi
exec "${TORCHRUN}" --standalone --nproc_per_node=2 \
    tools/o5replay/run_session.py "${REPLAY_ARGS[@]}" \
    --llm-cache 32768 \
    --attn-implementation sdpa \
    --generate-audio \
    --ls-mode explicit \
    --decode-mode greedy \
    --temperature 0.0 \
    --top-k 0 \
    --top-p 1.0 \
    --tts-temperature 0.8 \
    --tts-repetition-penalty 1.05 \
    --n-timesteps 10 \
    --experts-implementation hybrid \
    --grouped-prefill-min-tokens 100 \
    --strategy-hd \
    --strategy-hd-max-slice-nums 4 \
    --llm-graph \
    --tts-graph \
    --no-vocoder-graph \
    --tts-fast \
    --lmhead \
    --fuse-vision-audio \
    --batch-vision-feed
