#!/bin/bash
# Launch the model-serving backend in tp2 (2-card tensor-parallel) mode under torchrun SPMD.
# Both ranks build the TP-sharded LLM. Rank 0 drives serving/business logic;
# rank 1 waits in the model-provided LLM/graph worker loop and only participates
# in TP LLM compute.
#
# Prereqs: (1) extract the HF backbone once via tools/extract_backbone.py -> $O5_BACKBONE_DIR.
#          (2) config.json: model.deployment_mode="tp2", model.backbone_dir, model.llm_cache_len.
set -eo pipefail
export O5_DEPLOY_MODE=${O5_DEPLOY_MODE:-tp2}
export O5_BACKBONE_DIR=${O5_BACKBONE_DIR:?set to the extracted HF backbone dir}
export O5_LLM_CACHE=${O5_LLM_CACHE:-32768}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCHRUN=${TORCHRUN:-torchrun}
MASTER_ADDR=${O5_TORCHRUN_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${O5_TORCHRUN_MASTER_PORT:-${MASTER_PORT:-29500}}
# server.main() already handles SPMD: rank 0 serves HTTP, non-driver ranks run
# the model-provided worker_loop().
exec "$TORCHRUN" --nproc_per_node=2 --nnodes=1 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
     -m py_backend.server "$@"
