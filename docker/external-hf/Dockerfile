# Build context is assembled from frozen independent Git sources + hash-locked pip wheel inputs.
FROM python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370
RUN --network=none --mount=type=bind,source=apt-offline,target=/apt-inputs,readonly \
    --mount=type=bind,source=docker/external-hf/pip,target=/recipe,readonly \
    sh /recipe/install_debian_offline.sh /apt-inputs
ENV VIRTUAL_ENV=/opt/o5/.venv PATH=/opt/o5/.venv/bin:/usr/local/cuda-13.0/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-13.0 CUDA_INC_PATH=/opt/o5/.venv/lib/python3.10/site-packages/nvidia/cu13/include LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
    TORCH_CUDA_ARCH_LIST=12.0 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
RUN python -m venv /opt/o5/.venv
# Official source distributions supply compiler executables/headers; CUDA Python
# libraries and every Python distribution are installed from original wheels.
RUN --mount=type=bind,source=sources,target=/inputs,readonly \
    mkdir -p /usr/local/cuda-13.0 && \
    for archive in /inputs/cuda_*.tar.xz /inputs/libnvvm-*.tar.xz; do tar -xf "$archive" -C /usr/local/cuda-13.0 --strip-components=1; done
COPY requirements-install.lock /opt/o5/requirements-install.lock
RUN --mount=type=bind,source=wheelhouse,target=/wheels,readonly \
    python -m pip install --no-index --no-cache-dir --find-links=/wheels --no-deps --require-hashes -r /opt/o5/requirements-install.lock
RUN mkdir -p /usr/local/cuda-13.0/targets && \
    ln -s .. /usr/local/cuda-13.0/targets/x86_64-linux && \
    ln -s lib /usr/local/cuda-13.0/lib64 && \
    ln -sf /opt/o5/.venv/lib/python3.10/site-packages/nvidia/cu13/lib/libcublas.so.13 /usr/local/cuda-13.0/lib64/libcublas.so && \
    ln -sf /opt/o5/.venv/lib/python3.10/site-packages/nvidia/cu13/lib/libcublasLt.so.13 /usr/local/cuda-13.0/lib64/libcublasLt.so
# A real CPU compile catches incomplete CUDA redistribution assembly.
RUN printf '__global__ void o5_cuda_probe() {}\n' > /tmp/o5_cuda_probe.cu && \
    nvcc -std=c++20 -gencode arch=compute_120,code=sm_120 -c /tmp/o5_cuda_probe.cu -o /tmp/o5_cuda_probe.o && \
    rm /tmp/o5_cuda_probe.cu /tmp/o5_cuda_probe.o
RUN --mount=type=bind,source=materials,target=/materials,readonly \
    mkdir -p /opt/o5/hf /opt/o5/accel /opt/o5/demo && \
    tar -xf /materials/hf.tar -C /opt/o5/hf && \
    tar -xf /materials/accel-runtime.tar -C /opt/o5/accel && \
    tar -xf /materials/demo.tar -C /opt/o5/demo && \
    rm -rf /opt/o5/demo/static/mobile /opt/o5/demo/static/mobile-omni && \
    cp -a /materials/mobile /opt/o5/demo/static/mobile && \
    cp -a /materials/mobile-omni /opt/o5/demo/static/mobile-omni && \
    cp /materials/source-manifest.json /opt/o5/source-manifest.json
COPY docker/external-hf/ /opt/o5/package/
COPY installation-manifest.json /opt/o5/installation-manifest.json
RUN python /opt/o5/package/pip/audit_environment.py --output /opt/o5/environment-audit.json
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
RUN python /opt/o5/package/pip/cpu_validate.py
EXPOSE 8006 8007 22400 22500
ENTRYPOINT ["/opt/o5/.venv/bin/python", "/opt/o5/package/entrypoint.py"]
