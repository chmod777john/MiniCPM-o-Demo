#!/bin/bash
# MiniCPMO45 Service One-Click Environment Installation Script
#
# Usage:
#   bash install.sh
#   bash install.sh --with-accel
#
# Features:
#   1. Create a Python 3.10+ virtual environment
#   2. Install core dependencies from requirements.txt
#   3. Optionally install CUDA extension accelerators from requirements-accel.txt
#   4. Verify installation results
#
# Environment Variables (optional):
#   PYTHON=python3.10        Specify Python interpreter (default: python3.10)
#   MAX_JOBS=8               CUDA extension compilation parallelism (default: nproc)

set -e

# ============ Configuration ============

VENV_DIR=".venv"
PIP="${VENV_DIR}/bin/pip"
PYTHON_BIN="${VENV_DIR}/bin/python"
PYTHON="${PYTHON:-python3.10}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 8)}"
WITH_ACCEL=0

# ============ Colored Output ============

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

for arg in "$@"; do
    case "${arg}" in
        --with-accel)
            WITH_ACCEL=1
            ;;
        -h|--help)
            echo "Usage: bash install.sh [--with-accel]"
            echo "  --with-accel  Install optional CUDA extension accelerators into the same .venv"
            exit 0
            ;;
        *)
            error "Unknown argument: ${arg}"
            echo "Usage: bash install.sh [--with-accel]"
            exit 1
            ;;
    esac
done

# ============ Step 1: Create Virtual Environment ============

info "Step 1/4: Creating virtual environment (${VENV_DIR})"

if [ -d "${VENV_DIR}" ]; then
    warn "Virtual environment already exists: ${VENV_DIR}, skipping creation"
else
    if ! command -v "${PYTHON}" &> /dev/null; then
        error "${PYTHON} not found. Please install Python 3.10+ or specify the path via PYTHON=python3.x"
        exit 1
    fi

    PYTHON_VERSION=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "Using Python ${PYTHON_VERSION} (${PYTHON})"

    "${PYTHON}" -m venv "${VENV_DIR}"
    info "Virtual environment created successfully"
fi

${PIP} install --upgrade pip -q

# ============ Step 2: Install Core Dependencies ============

info "Step 2/4: Installing core dependencies (requirements.txt)"
${PIP} install -r requirements.txt
${PIP} install --no-deps "minicpmo-utils==1.0.6"
info "Core dependencies installed successfully"

# ============ Step 3: Optional CUDA Accelerators ============

if [ "${WITH_ACCEL}" = "1" ]; then
    info "Step 3/4: Installing optional CUDA accelerators (requirements-accel.txt)"
    warn "This step may download prebuilt wheels or compile CUDA extensions for this environment."
    MAX_JOBS=${MAX_JOBS} ${PIP} install --no-build-isolation -r requirements-accel.txt
    info "Optional CUDA accelerators installed successfully"
else
    info "Step 3/4: Skipping optional CUDA accelerators"
    warn "Use 'bash install.sh --with-accel' to add flash-attn and causal-conv1d to this same .venv."
fi

# ============ Step 4: Verify Installation ============

info "Step 4/4: Verifying installation"

# ============ Installation Summary ============

echo ""
echo "============================================"
info "Installation complete! Environment summary:"
echo "============================================"

${PYTHON_BIN} -c "
import torch
print(f'  Python:       {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.version.cuda}')
print(f'  GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

try:
    import flash_attn
    print(f'  Flash Attn:   {flash_attn.__version__} ✓')
    attn_backend = 'flash_attention_2'
except ImportError:
    print(f'  Flash Attn:   Not installed (will use SDPA)')
    attn_backend = 'sdpa'

import transformers
print(f'  Transformers: {transformers.__version__}')

try:
    import fla
    print(f'  FLA:          {fla.__version__} ✓')
except ImportError:
    print('  FLA:          Not installed')

try:
    import causal_conv1d
    print(f'  Causal Conv:  {causal_conv1d.__version__} ✓')
except ImportError:
    print('  Causal Conv:  Not installed')

print()
print(f'  Attention Backend: {attn_backend}')
"

echo ""
info "Next steps:"
echo "  1. Configure model path:"
echo "     cp config.example.json config.json"
echo "     # Edit config.json and set model.model_path"
echo ""
echo "  2. Start the service:"
echo "     bash start_all.sh"
echo ""
echo "  Optional acceleration:"
echo "     bash install.sh --with-accel"
echo "============================================"
