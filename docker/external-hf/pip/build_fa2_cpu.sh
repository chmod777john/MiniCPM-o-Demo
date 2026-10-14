#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/data/weihongliang/o5-dense-pip-build-20261014
export CUDA_VISIBLE_DEVICES=''
export MAX_JOBS=2 NVCC_THREADS=2
export CUDA_HOME=/usr/local/cuda-13.0
export CXX=/data/weihongliang/MiniCPM-o-Demo-wt-o5-omni-dev/.build-cu130/bin/x86_64-conda-linux-gnu-g++
export CC=/data/weihongliang/MiniCPM-o-Demo-wt-o5-omni-dev/.build-cu130/bin/x86_64-conda-linux-gnu-gcc
export PYTHONNOUSERSITE=1
export PATH="$TASK_ROOT/.venv/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin"
export TMPDIR="$TASK_ROOT/tmp-fa2"
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=120
export TORCH_CUDA_ARCH_LIST=12.0
mkdir -p "$TMPDIR" "$TASK_ROOT/sources" "$TASK_ROOT/native-wheels"
cd "$TASK_ROOT/sources"
test ! -e flash_attn-2.8.3.post1
tar -xf "$TASK_ROOT/flash_attn-2.8.3.post1.tar.gz"
cd flash_attn-2.8.3.post1
patch -p1 < "$TASK_ROOT/flash-attn-cxx20.patch"
"$CXX" --version
"$CUDA_HOME/bin/nvcc" --version
nice -n 10 python -m pip wheel --no-deps --no-build-isolation --no-cache-dir -v . --wheel-dir "$TASK_ROOT/native-wheels"
