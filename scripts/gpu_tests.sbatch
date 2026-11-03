#!/bin/bash
# GPU parity tests on one H100: the CUDA store path and flash-attn against the CPU references.
# Submit from the repo root: sbatch scripts/gpu_tests.sbatch
#
# FILL IN for the GT cluster (placeholders below are marked TODO):
#SBATCH --job-name=tinyserve-gpu-tests
#SBATCH --partition=TODO
#SBATCH --account=TODO
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Llama-only cluster: never point this job at Qwen or DeepSeek checkpoints.

set -euo pipefail
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# One-time environment on the cluster (uv keeps it fast). flash-attn compiles for a long time
# from source; prefer the prebuilt wheel matching this torch / CUDA / Python from its GitHub releases.
VENV=${VENV:-.venv-gpu}
if [ ! -x "$VENV/bin/python" ]; then
    uv venv "$VENV" --python 3.11
    uv pip install --python "$VENV/bin/python" -e . pytest triton
    uv pip install --python "$VENV/bin/python" flash-attn --no-build-isolation
fi
"$VENV/bin/python" -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'flash-attn', flash_attn.__version__)"

# The GPU tests first (they skip silently without CUDA, so fail loudly if they were skipped).
"$VENV/bin/python" -m pytest tests/test_gpu.py -v -rs | tee logs/gpu_tests_last.txt
! grep -q "SKIPPED" logs/gpu_tests_last.txt || { echo "GPU tests were skipped"; exit 1; }
# Then the whole suite on the GPU node, compiled kernels included.
"$VENV/bin/python" -m pytest tests -q
