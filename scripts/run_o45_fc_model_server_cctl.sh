#!/usr/bin/env bash
set -euo pipefail

DEMO_REPO=${DEMO_REPO:-/user/weihongliang/MiniCPM-o-Demo-o45-fc-try}
PYTHON_BIN=${PYTHON_BIN:-/user/weihongliang/MiniCPM-o-Demo/.venv/base/bin/python}
SDK_SRC=${SDK_SRC:-/user/weihongliang/o45_fc_assets/sdk/src}
MODEL_PATH=${MODEL_PATH:-/user/weihongliang/autoshow_omni/models/MiniCPM-o-4_5}
PT_PATH=${PT_PATH:-/user/weihongliang/o45_fc_assets/checkpoints/minicpm-v_100.pt}
REF_AUDIO_PATH=${REF_AUDIO_PATH:-/user/weihongliang/o45_fc_assets/training/delivery_train_data/media/system_reference/HTRef06.wav}
PORT=${PORT:-18081}

cd "${DEMO_REPO}"
export PYTHONPATH="${DEMO_REPO}:${SDK_SRC}:${PYTHONPATH:-}"

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "[O45_FC_MODEL_ENTRY] host=$(hostname) ip=${HOST_IP} demo_repo=${DEMO_REPO}"
echo "[MODEL_SERVER_READY_HINT] wait for uvicorn on port ${PORT}"

exec "${PYTHON_BIN}" -m audio_duplex_board.model_server \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model-path "${MODEL_PATH}" \
  --pt-path "${PT_PATH}" \
  --sdk-src "${SDK_SRC}" \
  --ref-audio-path "${REF_AUDIO_PATH}" \
  --attn-implementation sdpa
