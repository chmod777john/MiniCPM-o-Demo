#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR:-${SCRIPT_DIR}}"

export PROJECT_DIR="$PWD"
export VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
export MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
export PT_PATH="${PT_PATH:-/user/heweiquan/models/MiniCPM-o5/trained_model/20260724/job616069/iter_0000500_o5.pt}"
export BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/o5_backbones/job616069_iter0000500_hf}"

export O5_DEPLOY_MODE="${O5_DEPLOY_MODE:-tp2}"
export O5_ATTN_IMPLEMENTATION="${O5_ATTN_IMPLEMENTATION:-auto}"
export O5_LLM_CACHE="${O5_LLM_CACHE:-32768}"
export O5_LLM_GRAPH="${O5_LLM_GRAPH:-1}"
export O5_SPMD_HEARTBEAT_INTERVAL="${O5_SPMD_HEARTBEAT_INTERVAL:-30}"

export FC_BOARD_CASE_FOLDER="${FC_BOARD_CASE_FOLDER:-/user/weihongliang/o45_fc_assets/training/delivery_train_data}"

export GATEWAY_PORT="${GATEWAY_PORT:-8024}"
export GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8025}"
export BACKEND_PORT="${BACKEND_PORT:-22514}"
export WORKER_PORT="${WORKER_PORT:-22414}"
export WORKER_ID="${WORKER_ID:-o5-fc-board-job616069-worker}"
export WORKER_GPU_GROUP="${WORKER_GPU_GROUP:-cctl-a100-tp2-fc-board-job616069}"
export LOG_DIR="${LOG_DIR:-$PWD/run-logs/o5_tp2_fc_board_job616069_$(date +%Y%m%d_%H%M%S)}"

export FRPC_BIN="${FRPC_BIN:-/user/weihongliang/frp_0.65.0_linux_amd64/frpc}"
export FRPC_CONFIG="${FRPC_CONFIG:-}"
export ENABLE_FRP="${ENABLE_FRP:-0}"

bash scripts/start_o5_tp2_cctl_service.sh
