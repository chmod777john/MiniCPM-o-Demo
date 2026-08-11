#!/usr/bin/env bash
set -euo pipefail

# Continue one event-only ODB run with eight independent single_opt workers.
# This is data parallelism across eight one-GPU full-model processes, not TP2.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
HUMANEVALKIT_ROOT="${HUMANEVALKIT_ROOT:-/user/weihongliang/humanevalkit-codeup}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/user/weihongliang/o5_weights/joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000.pt}"
BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_weights/o5_backbone_hf_joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000}"
REF_AUDIO="${REF_AUDIO:-/backup/user/xubokai/humanevalkit_dev/migration_v2/humanevalkit/runs_chaoqun/HT_ref_audio.wav}"
DATA_ROOT="${DATA_ROOT:-/user/hechaoqun/final_data-v0}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to an existing run directory to resume}"
JUDGE_API_URL="${JUDGE_API_URL:-https://llm-center.modelbest.co/llm/v1/chat/completions}"
JUDGE_MODEL="${JUDGE_MODEL:-GEMINI_8daxh7}"
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
JUDGE_INFLIGHT_LIMIT="${JUDGE_INFLIGHT_LIMIT:-8}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
JUDGE_KEY_FILE="${JUDGE_KEY_FILE:-/user/weihongliang/lis.key}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-362}"
WORKER_COUNT=8

PYTHON="${VENV_DIR}/bin/python"
for path in "$PYTHON" "$HUMANEVALKIT_ROOT/src" "$MODEL_PATH" \
  "$CHECKPOINT_PATH" "$BACKBONE_DIR" "$REF_AUDIO" "$DATA_ROOT" "$JUDGE_KEY_FILE"; do
  if [ ! -e "$path" ]; then
    echo "[odb-demo-single-opt-8gpu] missing: $path" >&2
    exit 2
  fi
done

if [ -z "${JUDGE_API_KEY:-}" ]; then
  JUDGE_API_KEY="$(tr -d '\r\n' < "$JUDGE_KEY_FILE")"
fi
if [ -z "$JUDGE_API_KEY" ]; then
  echo "[odb-demo-single-opt-8gpu] empty Judge API key" >&2
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

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"
cd "$PROJECT_DIR"

base_limit=$((TOTAL_SAMPLES / WORKER_COUNT))
remainder=$((TOTAL_SAMPLES % WORKER_COUNT))
declare -a PIDS=()
declare -a LABELS=()

echo "[odb-demo-single-opt-8gpu] output=$OUTPUT_DIR samples=$TOTAL_SAMPLES"
echo "[odb-demo-single-opt-8gpu] deployment=8 independent single_opt workers; tp2=0"
echo "[odb-demo-single-opt-8gpu] acceleration=batched_mm llm_graph tts_graph tts_fast lmhead fuse_vision_audio; vocoder_graph=0"

cursor=0
for worker in $(seq 0 $((WORKER_COUNT - 1))); do
  limit="$base_limit"
  if (( worker < remainder )); then
    limit=$((limit + 1))
  fi
  start="$cursor"
  cursor=$((cursor + limit))
  log_file="$OUTPUT_DIR/logs/resume_worker${worker}.log"
  echo "[odb-demo-single-opt-8gpu] launch worker=$worker start=$start limit=$limit gpu=$worker"
  CUDA_VISIBLE_DEVICES="$worker" "$PYTHON" tools/o5eval/odb_tp2_demo_runner.py \
    --model-path "$MODEL_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --backbone-dir "$BACKBONE_DIR" \
    --ref-audio "$REF_AUDIO" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --start-index "$start" \
    --limit "$limit" \
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
    --resume \
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
    --seed 0 \
    >"$log_file" 2>&1 &
  PIDS+=("$!")
  LABELS+=("worker=$worker gpu=$worker range=$start:$limit")
done

rc=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[odb-demo-single-opt-8gpu] ${LABELS[$i]} completed"
  else
    echo "[odb-demo-single-opt-8gpu] ${LABELS[$i]} failed; see ${OUTPUT_DIR}/logs" >&2
    rc=1
  fi
done
exit "$rc"
