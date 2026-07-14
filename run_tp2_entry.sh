#!/usr/bin/env bash
set -euo pipefail

cd /user/weihongliang/MiniCPM-o-Demo-wt-o5-tp2-encapsulation-2026-07-14
export PROJECT_DIR=$PWD
export VENV_DIR=/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel
export MODEL_PATH=/user/weihongliang/MiniCPM-o-4_6
export PT_PATH=/user/caohao/training/omni-sft2-main-run/checkpoints/omni_sft2_main_run_iter1200.pt
export BACKBONE_DIR=/user/weihongliang/wangkaiqi/o5_backbone_hf
export O5_LLM_CACHE=32768
export O5_SPMD_HEARTBEAT_INTERVAL=30
export GATEWAY_PORT=8009
export GATEWAY_INTERNAL_PORT=8010
export BACKEND_PORT=22510
export WORKER_PORT=22410
export LOG_DIR=$PWD/run-logs/tp2_service_$(date +%Y%m%d_%H%M%S)
export FRPC_BIN=/user/weihongliang/frp_0.65.0_linux_amd64/frpc
export FRPC_CONFIG=/user/weihongliang/frp_0.65.0_linux_amd64/frpc_o5_tp2_8009_8445.toml
export ENABLE_FRP=1

bash scripts/start_o5_tp2_cctl_service.sh
