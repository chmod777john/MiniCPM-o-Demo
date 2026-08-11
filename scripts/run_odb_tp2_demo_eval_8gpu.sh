#!/usr/bin/env bash
set -euo pipefail

# One 8-GPU task hosts four independent 2-GPU TP2 ODB workers.  The model
# runner requires world size 2, so an 8-process torchrun is not equivalent.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
HUMANEVALKIT_ROOT="${HUMANEVALKIT_ROOT:-/user/weihongliang/humanevalkit-codeup}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/user/weihongliang/o5_weights/joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000.pt}"
BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_weights/o5_backbone_hf_joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000}"
REF_AUDIO="${REF_AUDIO:-/backup/user/xubokai/humanevalkit_dev/migration_v2/humanevalkit/runs_chaoqun/HT_ref_audio.wav}"
DATA_ROOT="${DATA_ROOT:-/user/hechaoqun/final_data-v0}"
OUT_ROOT="${OUT_ROOT:-/user/weihongliang/odb_eval_runs/odb_demo_tp2_iter4000_full_20260807}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29691}"
EVAL_SCOPE="${EVAL_SCOPE:-full}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-}"
GENERATE_AUDIO="${GENERATE_AUDIO:-0}"
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
JUDGE_API_URL="${JUDGE_API_URL:-https://llm-center.modelbest.co/llm/v1/chat/completions}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"

case "$EVAL_SCOPE" in
  full)
    TOTAL_SAMPLES="${TOTAL_SAMPLES:-662}"
    SCOPE_ARGS=()
    ;;
  event-only)
    TOTAL_SAMPLES="${TOTAL_SAMPLES:-362}"
    SCOPE_ARGS=(--event-only)
    ;;
  *)
    echo "[odb-demo-8gpu] EVAL_SCOPE must be full or event-only, got: $EVAL_SCOPE" >&2
    exit 2
    ;;
esac

PYTHON="${VENV_DIR}/bin/python"
TORCHRUN="${VENV_DIR}/bin/torchrun"
JUDGE_KEY_FILE="${JUDGE_KEY_FILE:-/user/weihongliang/lis.key}"

for path in "$PYTHON" "$TORCHRUN" "$HUMANEVALKIT_ROOT/src" "$MODEL_PATH" \
  "$CHECKPOINT_PATH" "$BACKBONE_DIR" "$REF_AUDIO" "$DATA_ROOT" "$JUDGE_KEY_FILE" \
  "$HUMANEVALKIT_ROOT/scripts/rejudge_from_results.py"; do
  if [ ! -e "$path" ]; then
    echo "[odb-demo-8gpu] missing: $path" >&2
    exit 2
  fi
done

if [ -z "${JUDGE_API_KEY:-}" ]; then
  JUDGE_API_KEY="$(tr -d '\r\n' < "$JUDGE_KEY_FILE")"
fi
if [ -z "$JUDGE_API_KEY" ]; then
  echo "[odb-demo-8gpu] empty Judge API key" >&2
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

mkdir -p "$OUT_ROOT" "$OUT_ROOT/logs"

echo "[odb-demo-8gpu] project=$PROJECT_DIR"
echo "[odb-demo-8gpu] benchmark=chaoqun_omn_bench scope=$EVAL_SCOPE samples=$TOTAL_SAMPLES"
echo "[odb-demo-8gpu] output=$OUT_ROOT"
echo "[odb-demo-8gpu] model=$MODEL_PATH"
echo "[odb-demo-8gpu] checkpoint=$CHECKPOINT_PATH"
echo "[odb-demo-8gpu] backbone=$BACKBONE_DIR"
echo "[odb-demo-8gpu] deployment=4x independent TP2 workers on GPU pairs 0-1,2-3,4-5,6-7"
echo "[odb-demo-8gpu] acceleration=tp2 sdpa batched_mm llm_graph tts_graph tts_fast lmhead fuse_vision_audio; vocoder_graph=0"
echo "[odb-demo-8gpu] generate_audio=$GENERATE_AUDIO"
echo "[odb-demo-8gpu] judge=GEMINI_8daxh7 workers=$JUDGE_WORKERS (post-inference)"

if [[ "$GENERATE_AUDIO" != "0" && "$GENERATE_AUDIO" != "1" ]]; then
  echo "[odb-demo-8gpu] GENERATE_AUDIO must be 0 or 1, got: $GENERATE_AUDIO" >&2
  exit 2
fi
if ! [[ "$JUDGE_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[odb-demo-8gpu] JUDGE_WORKERS must be a positive integer, got: $JUDGE_WORKERS" >&2
  exit 2
fi

if [ "$GENERATE_AUDIO" = "1" ]; then
  AUDIO_ARGS=(--generate-audio)
else
  AUDIO_ARGS=(--no-generate-audio)
fi

shard_count=4
base_limit=$((TOTAL_SAMPLES / shard_count))
remainder=$((TOTAL_SAMPLES % shard_count))
starts=()
limits=()
cursor=0
for ((shard = 0; shard < shard_count; shard++)); do
  shard_limit="$base_limit"
  if (( shard < remainder )); then
    shard_limit=$((shard_limit + 1))
  fi
  starts+=("$cursor")
  limits+=("$shard_limit")
  cursor=$((cursor + shard_limit))
done
gpu_pairs=("0,1" "2,3" "4,5" "6,7")
declare -a pids=()
declare -a labels=()

cd "$PROJECT_DIR"
for shard in 0 1 2 3; do
  shard_dir="$OUT_ROOT/shard${shard}"
  log_file="$OUT_ROOT/logs/shard${shard}.log"
  echo "[odb-demo-8gpu] launch shard=$shard start=${starts[$shard]} limit=${limits[$shard]} gpu=${gpu_pairs[$shard]}"
  CUDA_VISIBLE_DEVICES="${gpu_pairs[$shard]}" "$TORCHRUN" \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_addr=127.0.0.1 \
    --master_port="$((MASTER_PORT_BASE + shard))" \
    tools/o5eval/odb_tp2_demo_runner.py \
    --model-path "$MODEL_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --backbone-dir "$BACKBONE_DIR" \
    --ref-audio "$REF_AUDIO" \
    --data-root "$DATA_ROOT" \
    --output-dir "$shard_dir" \
    --start-index "${starts[$shard]}" \
    --limit "${limits[$shard]}" \
    "${SCOPE_ARGS[@]}" \
    "${AUDIO_ARGS[@]}" \
    --deployment-mode tp2 \
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
    --disable-judge \
    --judge-seed 0 \
    --seed 0 \
    >"$log_file" 2>&1 &
  pids+=("$!")
  labels+=("shard=$shard gpu=${gpu_pairs[$shard]}")
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[odb-demo-8gpu] ${labels[$i]} completed"
  else
    echo "[odb-demo-8gpu] ${labels[$i]} failed; see ${OUT_ROOT}/logs/shard${i}.log" >&2
    rc=1
  fi
done
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

# Keep model inference independent from remote Judge latency.  The rejudge
# helper rewrites each sample result, each shard report, and the merged report
# using a bounded thread pool, matching the Original job's Judge concurrency.
export HUMANEVALKIT_SRC="${HUMANEVALKIT_ROOT}/src"
"$PYTHON" "$HUMANEVALKIT_ROOT/scripts/rejudge_from_results.py" \
  --results-root "$OUT_ROOT" \
  --benchmark chaoqun_omn_bench \
  --judge-api-key "$JUDGE_API_KEY" \
  --judge-api-url "$JUDGE_API_URL" \
  --judge-model GEMINI_8daxh7 \
  --judge-max-tokens "$JUDGE_MAX_TOKENS" \
  --judge-temperature 0 \
  --judge-top-p 1 \
  --judge-max-retry 2 \
  --judge-sleep-between-retry 5 \
  --judge-seed 0 \
  --workers "$JUDGE_WORKERS" \
  --merge-shard-reports \
  --merge-base-dir "$OUT_ROOT" \
  --merged-name merged_report.json \
  --score-summary-file "$OUT_ROOT/score_summary.json"

echo "[odb-demo-8gpu] done: $OUT_ROOT/merged_report.json"
