#!/usr/bin/env bash
set -euo pipefail

# Run a small ODB split through the Demo TP2 backend.  The benchmark package
# itself stays outside this worktree; only its loader/schema/metrics are reused.

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
HUMANEVALKIT_ROOT="${HUMANEVALKIT_ROOT:-/user/weihongliang/humanevalkit-codeup}"
MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/user/weihongliang/o5_weights/joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000.pt}"
BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_weights/o5_backbone_hf_joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000}"
REF_AUDIO="${REF_AUDIO:-/backup/user/xubokai/humanevalkit_dev/migration_v2/humanevalkit/runs_chaoqun/HT_ref_audio.wav}"
DATA_ROOT="${DATA_ROOT:-/user/hechaoqun/final_data-v0}"
OUTPUT_DIR="${OUTPUT_DIR:-/user/weihongliang/odb_eval_runs/odb_demo_tp2_iter4000_smoke16}"
MASTER_PORT="${MASTER_PORT:-29681}"

PYTHON="${VENV_DIR}/bin/python"
TORCHRUN="${VENV_DIR}/bin/torchrun"

for path in "$PYTHON" "$TORCHRUN" "$HUMANEVALKIT_ROOT/src" "$MODEL_PATH" "$CHECKPOINT_PATH" "$BACKBONE_DIR" "$REF_AUDIO" "$DATA_ROOT"; do
  if [ ! -e "$path" ]; then
    echo "[odb-demo] missing: $path" >&2
    exit 2
  fi
done

export PROJECT_DIR HUMANEVALKIT_ROOT
export PYTHONPATH="${PROJECT_DIR}:${HUMANEVALKIT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[odb-demo] project=${PROJECT_DIR}"
echo "[odb-demo] benchmark=chaoqun_omn_bench limit=16"
echo "[odb-demo] model=${MODEL_PATH}"
echo "[odb-demo] checkpoint=${CHECKPOINT_PATH}"
echo "[odb-demo] backbone=${BACKBONE_DIR}"
echo "[odb-demo] data=${DATA_ROOT}"
echo "[odb-demo] output=${OUTPUT_DIR}"
echo "[odb-demo] deployment=tp2 experts=batched_mm llm_graph=1 tts_graph=1 vocoder_graph=${O5_VOCODER_GRAPH:-0}"

cd "${PROJECT_DIR}"
EXTRA_ARGS=()
if [ "${O5_VISION_BATCH:-0}" = "1" ]; then
  EXTRA_ARGS+=(--batch-vision-feed)
else
  EXTRA_ARGS+=(--no-batch-vision-feed)
fi
if [ "${O5_VOCODER_GRAPH:-0}" = "1" ]; then
  EXTRA_ARGS+=(--vocoder-graph)
else
  EXTRA_ARGS+=(--no-vocoder-graph)
fi
if [ -n "${JUDGE_API_KEY:-}" ]; then
  EXTRA_ARGS+=(--judge-api-key "${JUDGE_API_KEY}")
fi

exec "${TORCHRUN}" \
  --nproc_per_node=2 \
  --nnodes=1 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  tools/o5eval/odb_tp2_demo_runner.py \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --backbone-dir "${BACKBONE_DIR}" \
  --ref-audio "${REF_AUDIO}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit 16 \
  --generate-audio \
  --deployment-mode tp2 \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --experts-implementation "${O5_EXPERTS_IMPLEMENTATION:-batched_mm}" \
  --o5-llm-cache "${O5_LLM_CACHE:-65536}" \
  --llm-graph \
  --tts-graph \
  --tts-fast \
  --lmhead \
  --fuse-vision-audio \
  "${EXTRA_ARGS[@]}"
