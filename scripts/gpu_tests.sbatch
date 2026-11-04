#!/bin/bash
# GPU parity tests: the Triton store kernel and flash-attn against the CPU references.
# Run scripts/gpu_setup.sh once on a login node first, then from the repo root:
#   sbatch -p <gpu partition> scripts/gpu_tests.sbatch
# Cluster-specific options (partition, account, nodes to avoid) go on the command line, which
# overrides the defaults below.
#SBATCH --job-name=tinyserve-gpu-tests
#SBATCH --gres=gpu:h100:1                # any GPU works for the tests; gpu:1 takes the first free one
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source scripts/gpu_env.sh
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no environment at $VENV: run scripts/gpu_setup.sh on a login node first"; exit 1; }
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PY" -c "import torch, flash_attn; print('torch', torch.__version__, '| flash-attn', flash_attn.__version__)"

# The GPU tests skip silently without CUDA or flash-attn, so a skip here is a failure.
"$PY" -m pytest tests/test_gpu.py -v -rs | tee "logs/gpu_tests_${SLURM_JOB_ID}.txt"
if grep -q "SKIPPED" "logs/gpu_tests_${SLURM_JOB_ID}.txt"; then echo "GPU tests were skipped"; exit 1; fi
# Then everything else on the GPU node, with the compiled kernels on. The Qwen3 tests are left
# out: this job is for clusters where Qwen models may not run, even tiny random ones.
"$PY" -m pytest tests -q --ignore=tests/test_qwen3.py
