#!/bin/bash
# Launch the model-serving backend in tp2 (2-card tensor-parallel) mode under torchrun SPMD.
# Both ranks run the same code; rank 0 drives the gateway, rank 1 mirrors the backbone compute
# for the per-layer NCCL all_reduce. The deployment-mode framework (core.deploy) builds the
# TP-sharded model on each rank when O5_DEPLOY_MODE=tp2.
#
# A complete O5 safetensors bundle contains the Qwen config under ``llm/`` and
# the same root shards are reused for TP2.
set -eo pipefail
export O5_DEPLOY_MODE=${O5_DEPLOY_MODE:-tp2}
export O5_LLM_CACHE=${O5_LLM_CACHE:-32768}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCHRUN=${TORCHRUN:-torchrun}
MASTER_ADDR=${O5_TORCHRUN_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${O5_TORCHRUN_MASTER_PORT:-${MASTER_PORT:-29500}}
# server.main() already handles SPMD: rank 0 serves HTTP, non-driver ranks run worker_loop()
# (core/deploy/spmd.py) mirroring the driver s model calls. Validated live end-to-end.
exec "$TORCHRUN" --nproc_per_node=2 --nnodes=1 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
     -m py_backend.server "$@"
