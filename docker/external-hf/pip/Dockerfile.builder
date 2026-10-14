FROM python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370
RUN --network=none --mount=type=bind,source=apt-offline,target=/apt-inputs,readonly \
    --mount=type=bind,source=docker/external-hf/pip,target=/recipe,readonly \
    sh /recipe/install_debian_offline.sh /apt-inputs
# Official CUDA redistribution components, never restored from site-packages.
RUN --mount=type=bind,source=sources,target=/inputs,readonly \
    mkdir -p /usr/local/cuda-13.0 && \
    for archive in /inputs/cuda_*.tar.xz /inputs/libnvvm-*.tar.xz; do tar -xf "$archive" -C /usr/local/cuda-13.0 --strip-components=1; done
ENV VIRTUAL_ENV=/opt/o5/.venv PATH=/opt/o5/.venv/bin:/usr/local/cuda-13.0/bin:$PATH \
    CUDA_HOME=/usr/local/cuda-13.0 CUDA_INC_PATH=/opt/o5/.venv/lib/python3.10/site-packages/nvidia/cu13/include PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 MAX_JOBS=4 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST=12.0
RUN python -m venv /opt/o5/.venv
COPY install-public.lock /opt/o5/install-public.lock
RUN --mount=type=bind,source=wheelhouse,target=/wheels,readonly \
    python -m pip install --no-index --find-links=/wheels --no-deps --require-hashes -r /opt/o5/install-public.lock
RUN mkdir -p /usr/local/cuda-13.0/targets && \
    ln -s .. /usr/local/cuda-13.0/targets/x86_64-linux && \
    ln -s lib /usr/local/cuda-13.0/lib64 && \
    ln -sf /opt/o5/.venv/lib/python3.10/site-packages/nvidia/cu13/lib/libcublas.so.13 /usr/local/cuda-13.0/lib/libcublas.so && \
    ln -sf /opt/o5/.venv/lib/python3.10/site-packages/nvidia/cu13/lib/libcublasLt.so.13 /usr/local/cuda-13.0/lib/libcublasLt.so

# CPU-only compiler gate: verify the official split redistributions form a toolkit.
RUN printf '__global__ void o5_cuda_probe() {}\n' > /tmp/o5_cuda_probe.cu && \
    nvcc -std=c++20 -gencode arch=compute_120,code=sm_120 -c /tmp/o5_cuda_probe.cu -o /tmp/o5_cuda_probe.o && \
    rm /tmp/o5_cuda_probe.cu /tmp/o5_cuda_probe.o
