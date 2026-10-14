#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/data/weihongliang/o5-dense-pip-build-20261014
export CUDA_VISIBLE_DEVICES=''
export MAX_JOBS=2
export CUDA_HOME=/usr/local/cuda
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++
export PYTHONNOUSERSITE=1
export PYTHONPATH="$TASK_ROOT/accel/src"
export PATH="$TASK_ROOT/.venv/bin:/opt/rh/gcc-toolset-13/root/usr/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export TORCH_EXTENSIONS_DIR="$TASK_ROOT/extensions"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR"
python - <<'PY'
import sys,torch,json,sysconfig
print(json.dumps({'prefix':sys.prefix,'python':sys.version,'torch':torch.__version__,'torch_git':torch.version.git_version,'cuda':torch.version.cuda,'soabi':sysconfig.get_config_var('SOABI'),'visible_cuda':__import__('os').environ['CUDA_VISIBLE_DEVICES']},indent=2))
PY
nice -n 10 python -m minicpm_o_accel.llm.ops.dense_causal.build --torch-source /data/weihongliang/pro-public-torch-source-08187d9 --cutlass /data/weihongliang/hf-accel-pro-dense-causal-f781efe/cutlass-e05f953a5b3d38adc240df2ff928e0421c2abba3
