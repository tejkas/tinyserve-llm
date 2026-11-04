#!/bin/bash
# Real-weights sanity check (scripts/check_real_model.py) on one GPU. Download the checkpoint on a
# login node first, then from the repo root:
#   sbatch -p <gpu partition> scripts/check_real_model.sbatch /path/to/checkpoint
#SBATCH --job-name=tinyserve-real-model
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
export TERM=xterm-256color  # the submitting terminal's TERM may be unknown on the node
source scripts/gpu_env.sh
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no environment at $VENV: run scripts/gpu_setup.sh on a login node first"; exit 1; }
[ -f "${1:?usage: sbatch scripts/check_real_model.sbatch /path/to/checkpoint}/config.json" ] || { echo "no config.json in $1"; exit 1; }
export HF_HUB_OFFLINE=1  # the checkpoint is already local; compute nodes should not download
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" scripts/check_real_model.py "$1"
