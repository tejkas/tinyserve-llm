#!/bin/bash
# One-time setup on a rented single-GPU Linux machine without Slurm (e.g. a Lambda Labs H100).
# From the repo root:
#   bash scripts/lambda_setup.sh
# Builds two environments on the same torch and flash-attn: ours (~/venvs/tinyserve) and stock
# nano-vllm at a fixed commit (~/venvs/nano), and downloads Qwen3-0.6B. The Qwen trace is built
# elsewhere and copied to benchmarks/data/trace_qwen3.json (see benchmarks/build_trace.py).
set -euo pipefail
FLASH=2.8.3.post2
TRANSFORMERS=5.22.1
NANO_COMMIT=bb823b3e06983d71485a8e1f23715ebd87d98ef8

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
mkdir -p ~/venvs ~/models logs benchmarks/data

# torch 2.8 because flash-attn's newest prebuilt wheels stop there; anything newer compiles for hours.
make_env() {
    local py="$HOME/venvs/$1/bin/python"
    uv venv "$HOME/venvs/$1" --python 3.11
    uv pip install --python "$py" "torch==2.8.*" --index-url https://download.pytorch.org/whl/cu128
    local abi
    abi=$("$py" -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())")
    uv pip install --python "$py" \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH}/flash_attn-${FLASH}+cu12torch2.8cxx11abi${abi}-cp311-cp311-linux_x86_64.whl"
    uv pip install --python "$py" "transformers==${TRANSFORMERS}"
}

make_env tinyserve
uv pip install --python ~/venvs/tinyserve/bin/python -e . pytest

make_env nano
# --no-deps: nano-vllm lists flash-attn as a dependency, which would otherwise be compiled from source.
uv pip install --python ~/venvs/nano/bin/python --no-deps "git+https://github.com/GeeeekExplorer/nano-vllm.git@${NANO_COMMIT}"
uv pip install --python ~/venvs/nano/bin/python xxhash tqdm

~/venvs/tinyserve/bin/hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B

for env in tinyserve nano; do
    ~/venvs/$env/bin/python -c "import torch, flash_attn, transformers; print('$env: torch', torch.__version__, '| flash-attn', flash_attn.__version__, '| transformers', transformers.__version__)"
done
~/venvs/nano/bin/python -c "import nanovllm; print('nano-vllm imports')"
[ -f benchmarks/data/trace_qwen3.json ] || echo "next: copy the Qwen trace to benchmarks/data/trace_qwen3.json"
