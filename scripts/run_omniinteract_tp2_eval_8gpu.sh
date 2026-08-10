#!/usr/bin/env bash
set -euo pipefail

# Run four independent 2-GPU TP2 workers on one 8-GPU node and emit the
# standard OmniInteract leaf layout consumed by score.py.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
EVAL_PY="${EVAL_PY:-/user/houyueran/miniconda/envs/cpmo_tts_BM/bin/python}"
SUMMARY_PY="${SUMMARY_PY:-/user/houyueran/miniconda/envs/cpmo_35A3/bin/python}"
BENCH_DIR="${BENCH_DIR:-/user/houyueran/GIT/omni_TTS_Benchmark}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
PT_PATH="${PT_PATH:-/user/weihongliang/o5_weights/joint_stage2_40rank_alltry_best_fix_addNum_from_merge_equal_start_gemm_iter_3000.pt}"
BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_weights/o5_backbone_hf_joint_stage2_40rank_alltry_best_fix_addNum_from_merge_equal_start_gemm_iter_3000}"
PROMPT_WAV_ZH="${PROMPT_WAV_ZH:-/user/houyueran/GIT/SoulX-Podcast_MakeTrainData/ref_audio/ref_minicpm_signature.wav}"
PROMPT_WAV_EN="${PROMPT_WAV_EN:-/user/houyueran/GIT/SoulX-Podcast_MakeTrainData/ref_audio/F409-379_clip_310-324.wav}"
JOBS_FILE="${JOBS_FILE:-/user/houyueran/GIT/omni_TTS_Benchmark/OmniInteract_tts_eval/jobs_short60.tsv}"
OUT_ROOT="${OUT_ROOT:-/user/weihongliang/tts_eval_runs/omniinteract_demo_speeduptp2_alltry_iter3000}"
CKPT_NAME="${CKPT_NAME:-demo_speeduptp2_alltry_iter3000}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/_worker_logs}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29627}"
RUN_SCORE="${RUN_SCORE:-1}"
JOB_PART="${JOB_PART:-0}"
JOB_PARTS="${JOB_PARTS:-1}"

TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-100}"
TOP_P="${TOP_P:-0.8}"
TTS_TEMPERATURE="${TTS_TEMPERATURE:-0.8}"
TTS_REPETITION_PENALTY="${TTS_REPETITION_PENALTY:-1.05}"
TEXT_REPETITION_PENALTY="${TEXT_REPETITION_PENALTY:-1.05}"
TEXT_REPETITION_WINDOW_SIZE="${TEXT_REPETITION_WINDOW_SIZE:-512}"
LENGTH_PENALTY="${LENGTH_PENALTY:-1.0}"
FORCE_LISTEN_COUNT="${FORCE_LISTEN_COUNT:-0}"
MAX_NEW_SPEAK_TOKENS_PER_CHUNK="${MAX_NEW_SPEAK_TOKENS_PER_CHUNK:-20}"
N_TIMESTEPS="${N_TIMESTEPS:-10}"
O5_LLM_CACHE="${O5_LLM_CACHE:-65536}"

if (( JOB_PARTS < 1 || JOB_PART < 0 || JOB_PART >= JOB_PARTS )); then
  echo "[omniinteract-8gpu] invalid JOB_PART/JOB_PARTS: $JOB_PART/$JOB_PARTS" >&2
  exit 2
fi

JOB_INDEX_OFFSET=$((JOB_PART * 4))
JOB_STRIDE=$((JOB_PARTS * 4))

PYTHON="${VENV_DIR}/bin/python"
TORCHRUN="${VENV_DIR}/bin/torchrun"
for required in "$PYTHON" "$TORCHRUN" "$EVAL_PY" "$SUMMARY_PY" \
  "$PT_PATH" "$MODEL_PATH/config.json" "$BACKBONE_DIR" "$JOBS_FILE" \
  "$PROMPT_WAV_ZH" "$PROMPT_WAV_EN"; do
  if [ ! -e "$required" ]; then
    echo "[omniinteract-8gpu] missing: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUT_ROOT" "$LOG_DIR"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export O5_TTS_ARGMAX=0

ENGINE_LABEL="tp2 sdpa batched_mm llm_graph tts_graph tts_fast lmhead fuse_vision_audio batch_vision_feed; vocoder_graph=0"
ENGINE_FLAGS=(
  --llm-graph
  --tts-graph
  --no-vocoder-graph
  --tts-fast
  --lmhead
  --fuse-vision-audio
  --batch-vision-feed
)

echo "[omniinteract-8gpu] project=$PROJECT_DIR"
echo "[omniinteract-8gpu] jobs=$JOBS_FILE part=$JOB_PART/$JOB_PARTS stride=$JOB_STRIDE"
echo "[omniinteract-8gpu] out=$OUT_ROOT ckpt=$CKPT_NAME"
echo "[omniinteract-8gpu] model=$MODEL_PATH"
echo "[omniinteract-8gpu] pt=$PT_PATH"
echo "[omniinteract-8gpu] backbone=$BACKBONE_DIR"
echo "[omniinteract-8gpu] sampling=mode:sampling temperature:$TEMPERATURE top_k:$TOP_K top_p:$TOP_P tts_temperature:$TTS_TEMPERATURE length_penalty:$LENGTH_PENALTY"
echo "[omniinteract-8gpu] acceleration=$ENGINE_LABEL llm_cache=$O5_LLM_CACHE"

cd "$PROJECT_DIR"
declare -a PIDS=()
declare -a LABELS=()
declare -a GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")

for worker in "${!GPU_PAIRS[@]}"; do
  pair="${GPU_PAIRS[$worker]}"
  job_index=$((JOB_INDEX_OFFSET + worker))
  port=$((MASTER_PORT_BASE + worker))
  log="$LOG_DIR/worker_${worker}_gpu_${pair//,/}.log"
  echo "[omniinteract-8gpu] launch worker=$worker visible=$pair stride=$JOB_STRIDE log=$log"
  CUDA_VISIBLE_DEVICES="$pair" "$TORCHRUN" \
    --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" \
    tools/o5eval/omniinteract_tp2_offline_runner.py \
      --model-path "$MODEL_PATH" --ckpt-path "$PT_PATH" --backbone-dir "$BACKBONE_DIR" \
      --prompt-wav-zh "$PROMPT_WAV_ZH" --prompt-wav-en "$PROMPT_WAV_EN" \
      --jobs-file "$JOBS_FILE" --job-index "$job_index" --job-count 0 --job-stride "$JOB_STRIDE" \
      --out-root "$OUT_ROOT" --ckpt-name "$CKPT_NAME" \
      --deployment-mode tp2 --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
      --decode-mode sampling --temperature "$TEMPERATURE" --top-k "$TOP_K" --top-p "$TOP_P" \
      --tts-temperature "$TTS_TEMPERATURE" --tts-repetition-penalty "$TTS_REPETITION_PENALTY" \
      --text-repetition-penalty "$TEXT_REPETITION_PENALTY" \
      --text-repetition-window-size "$TEXT_REPETITION_WINDOW_SIZE" \
      --length-penalty "$LENGTH_PENALTY" --force-listen-count "$FORCE_LISTEN_COUNT" \
      --max-new-speak-tokens-per-chunk "$MAX_NEW_SPEAK_TOKENS_PER_CHUNK" \
      --n-timesteps "$N_TIMESTEPS" --experts-implementation "${O5_EXPERTS_IMPLEMENTATION:-batched_mm}" \
      --o5-llm-cache "$O5_LLM_CACHE" "${ENGINE_FLAGS[@]}" \
      --overwrite >"$log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("worker=$worker gpu=$pair")
done

rc=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[omniinteract-8gpu] ${LABELS[$i]} OK"
  else
    echo "[omniinteract-8gpu] ${LABELS[$i]} FAILED; see ${LOG_DIR}" >&2
    rc=1
  fi
done
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

if [ "$RUN_SCORE" = "1" ]; then
  cd "$BENCH_DIR"
  "$EVAL_PY" OmniInteract_tts_eval/score.py --out-root "$OUT_ROOT" --gpus 0
  "$SUMMARY_PY" OmniInteract_tts_eval/summarize.py --out-root "$OUT_ROOT"
fi
echo "[omniinteract-8gpu] done: $OUT_ROOT"
