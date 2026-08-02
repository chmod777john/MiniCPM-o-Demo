#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR:-${SCRIPT_DIR}}"
export PROJECT_DIR=$PWD
export VENV_DIR="${VENV_DIR:-/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel}"
export MODEL_PATH="${MODEL_PATH:-/user/weihongliang/MiniCPM-o-4_6}"
export PT_PATH="${PT_PATH:-/user/weihongliang/o5_weights/omni_sft2_main_run_iter1200.pt}"
export BACKBONE_DIR="${BACKBONE_DIR:-/user/weihongliang/wangkaiqi/o5_backbone_hf}"
export O5_DEPLOY_MODE="${O5_DEPLOY_MODE:-tp2}"
export O5_ATTN_IMPLEMENTATION="${O5_ATTN_IMPLEMENTATION:-auto}"
export O5_LLM_CACHE="${O5_LLM_CACHE:-32768}"
# Default to the narrowed TP2 path: rank 0 owns serving/duplex/TTS and the
# LLM graph runner synchronizes only the TP LLM work with rank 1.
export O5_LLM_GRAPH="${O5_LLM_GRAPH:-1}"
export O5_SPMD_HEARTBEAT_INTERVAL="${O5_SPMD_HEARTBEAT_INTERVAL:-30}"
export GATEWAY_PORT="${GATEWAY_PORT:-8011}"
export GATEWAY_INTERNAL_PORT="${GATEWAY_INTERNAL_PORT:-8012}"
export BACKEND_PORT="${BACKEND_PORT:-22512}"
export WORKER_PORT="${WORKER_PORT:-22412}"
export WORKER_ID="${WORKER_ID:-o5-tp2-llmgraph-narrow-worker}"
export WORKER_GPU_GROUP="${WORKER_GPU_GROUP:-cctl-a100-tp2-llmgraph-narrow}"
export O5_DETERMINISTIC_REPLAY="${O5_DETERMINISTIC_REPLAY:-0}"
export O5_SESSION_SEED="${O5_SESSION_SEED:-0}"
export O5_TTS_ARGMAX="${O5_TTS_ARGMAX:-1}"
export LOG_DIR="${LOG_DIR:-$PWD/run-logs/o5_tp2_llm_$(date +%Y%m%d_%H%M%S)}"
export FRPC_BIN="${FRPC_BIN:-/user/weihongliang/frp_0.65.0_linux_amd64/frpc}"
export FRPC_CONFIG="${FRPC_CONFIG:-/user/weihongliang/frp_0.65.0_linux_amd64/frpc_o5_tp2_llmgraph_narrow_8011_8458.toml}"
export ENABLE_FRP="${ENABLE_FRP:-0}"

bash scripts/start_o5_tp2_cctl_service.sh
