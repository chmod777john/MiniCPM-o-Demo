#!/bin/bash
# Launch the model-serving backend in tp2 (2-card tensor-parallel) mode under torchrun SPMD.
# Both ranks run the same code; rank 0 drives the gateway, rank 1 mirrors the backbone compute
# for the per-layer NCCL all_reduce. The deployment-mode framework (core.deploy) builds the
# TP-sharded model on each rank when O5_DEPLOY_MODE=tp2.
#
# Prereqs: (1) extract the HF backbone once via tools/extract_backbone.py -> \$O5_BACKBONE_DIR.
#          (2) config.json: model.deployment_mode=\"tp2\", model.backbone_dir, model.llm_cache_len.
set -eo pipefail
export O5_DEPLOY_MODE=tp2
export O5_BACKBONE_DIR=\${O5_BACKBONE_DIR:?set to the extracted HF backbone dir}
export O5_LLM_CACHE=\${O5_LLM_CACHE:-32768}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: the backend server entrypoint must, for SPMD, run the gateway I/O on the driver rank
# (core.deploy BuildResult.is_driver) and broadcast each duplex request input to all ranks via
# BuildResult.broadcast_input BEFORE running the unit (so both ranks run identical collectives).
exec torchrun --nproc_per_node=2 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 \
     -m py_backend.server \"\$@\"
