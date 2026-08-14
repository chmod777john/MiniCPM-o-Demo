#!/usr/bin/env bash
set -euo pipefail

# Run one static human-eval dataset with four independent TP2 workers.
# DATASET selects the physical list and output prefix used by the existing
# visualization pipeline: new_tigan_static -> new_static, old_tigan_v2 -> old_v2_all.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
PT_PATH="${PT_PATH:-/user/weihongliang/o5_weights/joint_stage2_40rank_alltry_best_fix_addNum_from_merge_equal_start_gemm_iter_3000.pt}"
BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_weights/o5_backbone_hf_joint_stage2_40rank_alltry_best_fix_addNum_from_merge_equal_start_gemm_iter_3000}"
REF_AUDIO="${REF_AUDIO:-/user/xubokai/audio_eval_3o/BH-Ref-HT-F224-Ref06_82_U001_话题_3_348s-355s.wav}"
SAVE_PATH="${SAVE_PATH:-/user/weihongliang/static_case_infer}"
DATASET="${DATASET:-new_tigan_static}"
CKPT_NAME="${CKPT_NAME:-$(basename "${PT_PATH%.pt}")}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29741}"
HF_MODULES_CACHE_ROOT="${HF_MODULES_CACHE_ROOT:-/tmp/o5_hf_modules_static_case_${CKPT_NAME}_$$}"
JOB_PART="${JOB_PART:-0}"
JOB_PARTS="${JOB_PARTS:-1}"

if (( JOB_PARTS < 1 || JOB_PART < 0 || JOB_PART >= JOB_PARTS )); then
  echo "invalid JOB_PART/JOB_PARTS: $JOB_PART/$JOB_PARTS" >&2
  exit 2
fi

LOCAL_WORKERS=4
GLOBAL_WORKER_COUNT=$((JOB_PARTS * LOCAL_WORKERS))
GLOBAL_WORKER_OFFSET=$((JOB_PART * LOCAL_WORKERS))

case "$DATASET" in
  new_tigan_static)
    VIDEO_LIST_DEFAULT="/user/wangchongyi/projects/visualization/omni_infer/video_paths_new.txt"
    PHYSICAL_SET="new_static"
    ;;
  old_tigan_v2)
    VIDEO_LIST_DEFAULT="/user/wangchongyi/projects/visualization/omni_infer/video_paths_old.txt"
    PHYSICAL_SET="old_v2_all"
    ;;
  *)
    echo "unsupported DATASET=$DATASET; expected new_tigan_static or old_tigan_v2" >&2
    exit 2
    ;;
esac

VIDEO_LIST="${VIDEO_LIST:-$VIDEO_LIST_DEFAULT}"
PREFIX_DIR="${PREFIX_DIR:-${PHYSICAL_SET}/${CKPT_NAME}}"
LOG_DIR="${LOG_DIR:-${SAVE_PATH}/${CKPT_NAME}/_logs/${PHYSICAL_SET}/part_${JOB_PART}_of_${JOB_PARTS}}"
PYTHON="${VENV_DIR}/bin/python"
TORCHRUN="${VENV_DIR}/bin/torchrun"

for required in "$PYTHON" "$TORCHRUN" "$MODEL_PATH/config.json" "$PT_PATH" \
  "$BACKBONE_DIR" "$REF_AUDIO" "$VIDEO_LIST"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR" "$HF_MODULES_CACHE_ROOT"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export O5_TTS_ARGMAX="${O5_TTS_ARGMAX:-0}"

TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-100}"
TOP_P="${TOP_P:-0.8}"
TTS_TEMPERATURE="${TTS_TEMPERATURE:-0.8}"
TTS_REPETITION_PENALTY="${TTS_REPETITION_PENALTY:-1.10}"
TEXT_REPETITION_PENALTY="${TEXT_REPETITION_PENALTY:-1.05}"
TEXT_REPETITION_WINDOW_SIZE="${TEXT_REPETITION_WINDOW_SIZE:-512}"
LENGTH_PENALTY="${LENGTH_PENALTY:-1.0}"
FORCE_LISTEN_COUNT="${FORCE_LISTEN_COUNT:-0}"
MAX_NEW_SPEAK_TOKENS_PER_CHUNK="${MAX_NEW_SPEAK_TOKENS_PER_CHUNK:-20}"
O5_LLM_CACHE="${O5_LLM_CACHE:-65536}"
SEED="${SEED:-1234}"
SLICE_NUMS="${SLICE_NUMS:-1}"

echo "[static-case] project=$PROJECT_DIR"
echo "[static-case] dataset=$DATASET videos=$VIDEO_LIST"
echo "[static-case] output=$SAVE_PATH/$CKPT_NAME/$PREFIX_DIR"
echo "[static-case] model=$MODEL_PATH"
echo "[static-case] checkpoint=$PT_PATH"
echo "[static-case] backbone=$BACKBONE_DIR"
echo "[static-case] job_part=$JOB_PART/$JOB_PARTS global_workers=$GLOBAL_WORKER_COUNT"
echo "[static-case] topology=$LOCAL_WORKERS independent TP2 workers on this 8-GPU node"

cd "$PROJECT_DIR"
declare -a PIDS=()
declare -a LABELS=()
declare -a GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")

for worker in "${!GPU_PAIRS[@]}"; do
  pair="${GPU_PAIRS[$worker]}"
  global_worker=$((GLOBAL_WORKER_OFFSET + worker))
  port=$((MASTER_PORT_BASE + worker))
  log="$LOG_DIR/worker_${global_worker}_gpu_${pair//,/}.log"
  hf_modules_cache="$HF_MODULES_CACHE_ROOT/$worker"
  mkdir -p "$hf_modules_cache"
  echo "[static-case] launch local_worker=$worker global_worker=$global_worker visible=$pair log=$log"
  HF_MODULES_CACHE="$hf_modules_cache" CUDA_VISIBLE_DEVICES="$pair" "$TORCHRUN" \
    --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" \
    tools/o5eval/static_case_tp2_offline_runner.py \
      --model-path "$MODEL_PATH" --ckpt-path "$PT_PATH" --backbone-dir "$BACKBONE_DIR" \
      --ref-audio "$REF_AUDIO" --video-list "$VIDEO_LIST" \
      --save-path "$SAVE_PATH" --ckpt-name "$CKPT_NAME" --prefix-dir "$PREFIX_DIR" \
      --worker-index "$global_worker" --worker-count "$GLOBAL_WORKER_COUNT" --run-index 1 --resume \
      --seed "$SEED" --slice-nums "$SLICE_NUMS" --chunk-ms 1000 \
      --deployment-mode tp2 --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
      --decode-mode sampling --temperature "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P" \
      --tts-temperature "$TTS_TEMPERATURE" --tts-repetition-penalty "$TTS_REPETITION_PENALTY" \
      --text-repetition-penalty "$TEXT_REPETITION_PENALTY" \
      --text-repetition-window-size "$TEXT_REPETITION_WINDOW_SIZE" \
      --length-penalty "$LENGTH_PENALTY" --force-listen-count "$FORCE_LISTEN_COUNT" \
      --max-new-speak-tokens-per-chunk "$MAX_NEW_SPEAK_TOKENS_PER_CHUNK" \
      --experts-implementation "${O5_EXPERTS_IMPLEMENTATION:-batched_mm}" \
      --o5-llm-cache "$O5_LLM_CACHE" --llm-graph --tts-graph --no-vocoder-graph \
      --tts-fast --lmhead --fuse-vision-audio --batch-vision-feed \
      >"$log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("worker=$global_worker gpu=$pair")
done

rc=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[static-case] ${LABELS[$index]} OK"
  else
    echo "[static-case] ${LABELS[$index]} FAILED; see $LOG_DIR" >&2
    rc=1
  fi
done

if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi
echo "[static-case] done: $SAVE_PATH/$CKPT_NAME/$PREFIX_DIR"
