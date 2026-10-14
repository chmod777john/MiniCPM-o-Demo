#!/usr/bin/env bash
set -euo pipefail
: "${TASK_ROOT:?Set TASK_ROOT to the isolated native build directory}"
: "${CXX:?Set CXX to the recorded GCC compiler}"
: "${CC:?Set CC to the recorded C compiler}"
: "${CUDA_HOME:?Set CUDA_HOME to the pinned CUDA 13.0.88 toolkit}"
export CUDA_VISIBLE_DEVICES=''
export MAX_JOBS=1 NVCC_THREADS=2
export PYTHONNOUSERSITE=1
export PATH="$TASK_ROOT/.venv/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export TMPDIR="$TASK_ROOT/tmp-causal"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST=12.0
mkdir -p "$TMPDIR" "$TASK_ROOT/sources" "$TASK_ROOT/native-wheels"
cd "$TASK_ROOT/sources"
test ! -e causal_conv1d-1.7.0
tar -xf "$TASK_ROOT/causal_conv1d-1.7.0.tar.gz"
cd causal_conv1d-1.7.0
# Upstream setup fixes its own multi-arch list and nvcc --threads 4. Preserve
# every architecture/kernel/flag; MAX_JOBS=1 caps it at one four-thread nvcc.
"$CXX" --version
"$CUDA_HOME/bin/nvcc" --version
nice -n 10 python -m pip wheel --no-deps --no-build-isolation --no-cache-dir -v . --wheel-dir "$TASK_ROOT/native-wheels"
