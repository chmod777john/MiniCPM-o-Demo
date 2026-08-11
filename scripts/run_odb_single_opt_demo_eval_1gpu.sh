#!/usr/bin/env bash
set -euo pipefail

# Run the ODB event-oriented split with one full model on one GPU.  This keeps
# the Demo optimization engine enabled while making the TP2 dimension explicit.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
HUMANEVALKIT_ROOT="${HUMANEVALKIT_ROOT:-/user/weihongliang/humanevalkit-codeup}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/user/weihongliang/o5_weights/joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000.pt}"
BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_weights/o5_backbone_hf_joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000}"
REF_AUDIO="${REF_AUDIO:-/backup/user/xubokai/humanevalkit_dev/migration_v2/humanevalkit/runs_chaoqun/HT_ref_audio.wav}"
DATA_ROOT="${DATA_ROOT:-/user/hechaoqun/final_data-v0}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to an isolated run directory}"
JUDGE_API_URL="${JUDGE_API_URL:-https://llm-center.modelbest.co/llm/v1/chat/completions}"
JUDGE_MODEL="${JUDGE_MODEL:-GEMINI_8daxh7}"
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
JUDGE_INFLIGHT_LIMIT="${JUDGE_INFLIGHT_LIMIT:-8}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
JUDGE_KEY_FILE="${JUDGE_KEY_FILE:-/user/weihongliang/lis.key}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-362}"

PYTHON="${VENV_DIR}/bin/python"

for path in "$PYTHON" "$HUMANEVALKIT_ROOT/src" "$MODEL_PATH" \
  "$CHECKPOINT_PATH" "$BACKBONE_DIR" "$REF_AUDIO" "$DATA_ROOT" "$JUDGE_KEY_FILE"; do
  if [ ! -e "$path" ]; then
    echo "[odb-demo-single-opt] missing: $path" >&2
    exit 2
  fi
done

if [ -z "${JUDGE_API_KEY:-}" ]; then
  JUDGE_API_KEY="$(tr -d '\r\n' < "$JUDGE_KEY_FILE")"
fi
if [ -z "$JUDGE_API_KEY" ]; then
  echo "[odb-demo-single-opt] empty Judge API key" >&2
  exit 2
fi
if ! [[ "$TOTAL_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[odb-demo-single-opt] TOTAL_SAMPLES must be positive, got: $TOTAL_SAMPLES" >&2
  exit 2
fi

export PROJECT_DIR HUMANEVALKIT_ROOT
export PYTHONPATH="${PROJECT_DIR}:${HUMANEVALKIT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export http_proxy="${http_proxy:-http://whitelist-proxy.cybertron.svc.cluster.local:7891}"
export https_proxy="${https_proxy:-http://whitelist-proxy.cybertron.svc.cluster.local:7891}"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

echo "[odb-demo-single-opt] project=$PROJECT_DIR"
echo "[odb-demo-single-opt] benchmark=chaoqun_omn_bench scope=event-only samples=$TOTAL_SAMPLES"
echo "[odb-demo-single-opt] output=$OUTPUT_DIR"
echo "[odb-demo-single-opt] model=$MODEL_PATH"
echo "[odb-demo-single-opt] checkpoint=$CHECKPOINT_PATH"
echo "[odb-demo-single-opt] backbone=$BACKBONE_DIR"
echo "[odb-demo-single-opt] deployment=single_opt gpu=1"
echo "[odb-demo-single-opt] acceleration=batched_mm llm_graph tts_graph tts_fast lmhead fuse_vision_audio; vocoder_graph=0"
echo "[odb-demo-single-opt] judge=model=$JUDGE_MODEL workers=$JUDGE_WORKERS inflight=$JUDGE_INFLIGHT_LIMIT"

exec "$PYTHON" tools/o5eval/odb_tp2_demo_runner.py \
  --model-path "$MODEL_PATH" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --backbone-dir "$BACKBONE_DIR" \
  --ref-audio "$REF_AUDIO" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$TOTAL_SAMPLES" \
  --event-only \
  --generate-audio \
  --deployment-mode single_opt \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --experts-implementation "${O5_EXPERTS_IMPLEMENTATION:-batched_mm}" \
  --o5-llm-cache "${O5_LLM_CACHE:-65536}" \
  --llm-graph \
  --tts-graph \
  --tts-fast \
  --lmhead \
  --fuse-vision-audio \
  --no-batch-vision-feed \
  --no-vocoder-graph \
  --judge-api-key "$JUDGE_API_KEY" \
  --judge-api-url "$JUDGE_API_URL" \
  --judge-model "$JUDGE_MODEL" \
  --judge-max-tokens "$JUDGE_MAX_TOKENS" \
  --judge-temperature 0 \
  --judge-top-p 1 \
  --judge-max-retry 2 \
  --judge-sleep-between-retry 5 \
  --judge-seed 0 \
  --judge-concurrency "$JUDGE_WORKERS" \
  --judge-inflight-limit "$JUDGE_INFLIGHT_LIMIT" \
  --seed 0
