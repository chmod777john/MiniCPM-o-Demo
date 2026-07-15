#!/usr/bin/env bash
set -euo pipefail

cd /user/weihongliang/MiniCPM-o-Demo-wt-o5-tp2-llm-wrapper-2026-07-15
export PROJECT_DIR=$PWD
export VENV_DIR=/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel
export MODEL_PATH=/user/weihongliang/MiniCPM-o-4_6
export PT_PATH=/user/weihongliang/o5_weights/omni_sft2_main_run_iter1200.pt
export BACKBONE_DIR=/user/weihongliang/wangkaiqi/o5_backbone_hf
export O5_DEPLOY_MODE=tp2_llm
export O5_LLM_CACHE=32768
export O5_SPMD_HEARTBEAT_INTERVAL=30
export GATEWAY_PORT=8009
export GATEWAY_INTERNAL_PORT=8010
export BACKEND_PORT=22510
export WORKER_PORT=22410
export WORKER_ID=o5-tp2-llm-wrapper-worker
export WORKER_GPU_GROUP=cctl-a100-tp2-llm
export LOG_DIR=$PWD/run-logs/o5_tp2_llm_$(date +%Y%m%d_%H%M%S)
export FRPC_BIN=/user/weihongliang/frp_0.65.0_linux_amd64/frpc
export FRPC_CONFIG=/user/weihongliang/frp_0.65.0_linux_amd64/frpc_o5_tp2_llm_8009_8447.toml
export ENABLE_FRP=1

bash scripts/start_o5_tp2_cctl_service.sh
