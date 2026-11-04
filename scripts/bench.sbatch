#!/bin/bash
# A benchmark grid on one GPU: everything after the script name is passed to benchmarks.driver.
#   sbatch -p <gpu partition> scripts/bench.sbatch --engine ours --model /path/to/checkpoint \
#       --trace benchmarks/data/trace_llama.json --conversations 300 --out-dir benchmarks/results/llama
# Finished runs are skipped, so resubmitting the same command resumes a grid cut short by the time limit.
#SBATCH --job-name=tinyserve-bench
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export TERM=xterm-256color  # the submitting terminal's TERM may be unknown on the node
source scripts/gpu_env.sh
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no environment at $VENV: run scripts/gpu_setup.sh on a login node first"; exit 1; }
export HF_HUB_OFFLINE=1  # checkpoints are local; compute nodes should not download
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -m benchmarks.driver --python "$PY" "$@"
