# syntax=docker/dockerfile:1.7
# Build context is assembled from frozen independent Git sources + verified Pro5000 env.
FROM python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgomp1 curl ca-certificates g++ make git procps \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/o5/.venv PATH=/opt/o5/.venv/bin:/usr/local/cuda-13.0/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-13.0 LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
    TORCH_CUDA_ARCH_LIST=12.0 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
RUN python -m venv /opt/o5/.venv
RUN --mount=type=bind,source=runtime,target=/runtime,readonly \
    tar -xzf /runtime/pro5000-python310-site.tar.gz -C /opt/o5/.venv && \
    mkdir -p /usr/local/cuda-13.0 && \
    tar -xzf /runtime/cuda13-compiler.tar.gz -C /usr/local/cuda-13.0 && \
    /opt/o5/.venv/bin/python -m ensurepip
# utils metadata includes old optional dependency pins; frozen runtime wins.
RUN --mount=type=bind,source=wheels,target=/wheels,readonly \
    python -m pip install --no-deps /wheels/minicpmo_utils-1.0.7-py3-none-any.whl /wheels/minicpm_o5_sdk-0.0.6-py3-none-any.whl
RUN --mount=type=bind,source=materials,target=/materials,readonly \
    mkdir -p /opt/o5/hf /opt/o5/accel /opt/o5/demo && \
    tar -xf /materials/hf.tar -C /opt/o5/hf && \
    tar -xf /materials/accel-runtime.tar -C /opt/o5/accel && \
    tar -xf /materials/demo.tar -C /opt/o5/demo && \
    rm -rf /opt/o5/demo/static/mobile /opt/o5/demo/static/mobile-omni && \
    cp -a /materials/mobile /opt/o5/demo/static/mobile && \
    cp -a /materials/mobile-omni /opt/o5/demo/static/mobile-omni && \
    cp /materials/source-manifest.json /opt/o5/source-manifest.json
COPY tools/ninja /opt/o5/.venv/bin/ninja
RUN chmod +x /opt/o5/.venv/bin/ninja
COPY docker/external-hf/ /opt/o5/package/
ENV PROJECT_DIR=/opt/o5/demo O5_CANONICAL_PATH=/opt/o5/hf \
    O5_ACCELERATION_PATH=/opt/o5/accel O5_ACCELERATION_HARDWARE=pro5000 \
    O5_DEPLOY_MODE=tp2 O5_MOE_PARALLEL=tp O5_ATTN_IMPLEMENTATION=sdpa \
    DUPLEX_SLOT_PROTOCOL=1 O5_LLM_GRAPH=1 O5_TTS_GRAPH=1 O5_VOCODER_GRAPH=1 \
    CANONICAL_UNIT_PREFILL_BATCH=1 O5_COMPACT_LOGITS=1 \
    O5_TTS_FA2_WHOLE_DECODE=1 O5_CANVAS_WINDOW_EXACT_CHAIN=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
    MODEL_PATH=/models/o5 O5_ROLE=worker GATEWAY_PORT=8006 GATEWAY_INTERNAL_PORT=8007 \
    BACKEND_PORT=22500 WORKER_PORT=22400
WORKDIR /opt/o5/demo
EXPOSE 8006 8007 22400 22500
ENTRYPOINT ["/opt/o5/.venv/bin/python", "/opt/o5/package/entrypoint.py"]
