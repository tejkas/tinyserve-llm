#!/bin/bash
# One-time GPU environment, run on a cluster login node (it downloads ~3 GB; compute nodes should not).
#   bash scripts/gpu_setup.sh
#
# torch is pinned to 2.8 because flash-attn's newest prebuilt wheels (v2.8.3.post2) stop there;
# a newer torch would make flash-attn compile from source, which takes hours.
set -euo pipefail
source "$(dirname "$0")/gpu_env.sh"
FLASH=2.8.3.post2

if ! command -v uv >/dev/null; then
    echo "uv not found. Install it into your home directory (no admin needed), then rerun:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
mkdir -p "$(dirname "$VENV")" "$UV_CACHE_DIR"
mkdir -p logs  # sbatch --output needs it to exist before the job starts
uv venv "$VENV" --python 3.11  # uv fetches Python 3.11 itself; no module load needed
PY="$VENV/bin/python"
uv pip install --python "$PY" "torch==2.8.*" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" -e . pytest

# Pick the flash-attn wheel matching this torch's C++ ABI.
ABI=$("$PY" -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())")
WHEEL="flash_attn-${FLASH}+cu12torch2.8cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"
uv pip install --python "$PY" "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH}/${WHEEL}"

"$PY" -c "import torch, triton, flash_attn; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| triton', triton.__version__, '| flash-attn', flash_attn.__version__)"
