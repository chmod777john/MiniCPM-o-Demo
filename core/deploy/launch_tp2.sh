#!/bin/bash
# Launch the model-serving backend in tp2 (2-card tensor-parallel) mode under torchrun SPMD.
# Both ranks build the TP-sharded LLM. Rank 0 drives serving/business logic;
# rank 1 waits in the model-provided LLM/graph worker loop and only participates
# in TP LLM compute.
#
# A complete O5 safetensors bundle contains the Qwen config under ``llm/`` and
# the same root shards are reused for TP2.
set -eo pipefail
export O5_DEPLOY_MODE=${O5_DEPLOY_MODE:-tp2}
export O5_LLM_CACHE=${O5_LLM_CACHE:-32768}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=${PYTHON:-python}
MASTER_ADDR=${O5_TORCHRUN_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${O5_TORCHRUN_MASTER_PORT:-${MASTER_PORT:-29500}}
# server.main() already handles SPMD: rank 0 serves HTTP, non-driver ranks run
# the model-provided worker_loop().
if [ -n "${TORCHRUN:-}" ]; then
  exec "${TORCHRUN}" --nproc_per_node=2 --nnodes=1 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
       -m py_backend.server "$@"
fi
exec "${PYTHON}" -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
     --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
     -m py_backend.server "$@"
