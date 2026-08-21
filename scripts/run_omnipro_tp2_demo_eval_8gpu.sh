#!/usr/bin/env bash
set -euo pipefail

# Run four independent 2-GPU TP2 Demo workers on one 8-GPU node.
# Inference writes result.json only; remote OmniPro judging is a separate step.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
HUMANEVALKIT_ROOT="${HUMANEVALKIT_ROOT:-/user/weihongliang/humanevalkit-codeup}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?set CHECKPOINT_PATH}"
BACKBONE_DIR="${BACKBONE_DIR:?set BACKBONE_DIR}"
DATA_ROOT="${DATA_ROOT:-/user/sunyinuo/data/OmniPro}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-2}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-2700}"
PORT_BASE="${PORT_BASE:-29731}"
LIMIT="${LIMIT:-$TOTAL_SAMPLES}"
RESUME="${RESUME:-0}"

PYTHON="${VENV_DIR}/bin/python"
TORCHRUN="${VENV_DIR}/bin/torchrun"

[[ -x "$PYTHON" ]] || { echo "missing python: $PYTHON" >&2; exit 2; }
[[ -x "$TORCHRUN" ]] || { echo "missing torchrun: $TORCHRUN" >&2; exit 2; }
for path in "$HUMANEVALKIT_ROOT/src" "$MODEL_PATH" "$CHECKPOINT_PATH" "$BACKBONE_DIR" "$DATA_ROOT"; do
  [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 2; }
done

IFS=',' read -ra CARDS <<< "$GPUS"
N=${#CARDS[@]}
(( TP_SIZE >= 2 )) || { echo "TP_SIZE must be >= 2" >&2; exit 2; }
(( N % TP_SIZE == 0 )) || { echo "GPU count must be divisible by TP_SIZE" >&2; exit 2; }
NGROUPS=$((N / TP_SIZE))

mkdir -p "$OUT_ROOT/logs"
export PROJECT_DIR HUMANEVALKIT_ROOT
export PYTHONPATH="${PROJECT_DIR}:${HUMANEVALKIT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[omnipro-demo-8gpu] project=$PROJECT_DIR"
echo "[omnipro-demo-8gpu] data=$DATA_ROOT samples=$LIMIT"
echo "[omnipro-demo-8gpu] output=$OUT_ROOT"
echo "[omnipro-demo-8gpu] model=$MODEL_PATH"
echo "[omnipro-demo-8gpu] checkpoint=$CHECKPOINT_PATH"
echo "[omnipro-demo-8gpu] backbone=$BACKBONE_DIR"
echo "[omnipro-demo-8gpu] workers=$NGROUPS TP_SIZE=$TP_SIZE GPUs=$GPUS"
echo "[omnipro-demo-8gpu] max_new_speak_tokens=${MAX_NEW_SPEAK_TOKENS_PER_CHUNK:-1024}"

IFS=',' read -ra CARD_LIST <<< "$GPUS"
pids=()
groups=()
for ((group=0; group<NGROUPS; group++)); do
  visible=""
  for ((offset=0; offset<TP_SIZE; offset++)); do
    card="${CARD_LIST[$((group * TP_SIZE + offset))]}"
    visible="${visible:+$visible,}${card}"
  done
  port=$((PORT_BASE + group))
  extra=()
  [[ "$RESUME" == "1" ]] && extra+=(--resume)
  echo "[omnipro-demo-8gpu] launch group=$group gpu=$visible port=$port"
  CUDA_VISIBLE_DEVICES="$visible" "$TORCHRUN" \
    --nproc_per_node="$TP_SIZE" \
    --nnodes=1 \
    --master_addr=127.0.0.1 \
    --master_port="$port" \
    tools/o5eval/omnipro_tp2_demo_runner.py \
    --model-path "$MODEL_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --backbone-dir "$BACKBONE_DIR" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT_ROOT/shard${group}" \
    --limit "$LIMIT" \
    --num-shards "$NGROUPS" \
    --shard-index "$group" \
    --deployment-mode "${DEPLOYMENT_MODE:-tp2}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
    --experts-implementation "${O5_EXPERTS_IMPLEMENTATION:-batched_mm}" \
    --o5-llm-cache "${O5_LLM_CACHE:-65536}" \
    --max-new-speak-tokens-per-chunk "${MAX_NEW_SPEAK_TOKENS_PER_CHUNK:-1024}" \
    --seed "${SEED:-0}" \
    --decode-mode "${DECODE_MODE:-sampling}" \
    --temperature "${TEMPERATURE:-0.7}" \
    --top-k "${TOP_K:-100}" \
    --top-p "${TOP_P:-0.8}" \
    --llm-graph \
    --tts-graph \
    --tts-fast \
    --lmhead \
    --fuse-vision-audio \
    --no-batch-vision-feed \
    --no-vocoder-graph \
    "${extra[@]}" \
    >"$OUT_ROOT/logs/group${group}.log" 2>&1 &
  pids+=("$!")
  groups+=("$group")
done

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[omnipro-demo-8gpu] group=${groups[$i]} completed"
  else
    echo "[omnipro-demo-8gpu] group=${groups[$i]} failed; see $OUT_ROOT/logs/group${groups[$i]}.log" >&2
    rc=1
  fi
done

if [[ "$rc" -eq 0 && -f "$HUMANEVALKIT_ROOT/scripts/merge_shard_reports.py" ]]; then
  "$PYTHON" "$HUMANEVALKIT_ROOT/scripts/merge_shard_reports.py" \
    --output-dir "$OUT_ROOT" \
    --merged-name merged_report.json
fi
exit "$rc"
