#!/bin/bash
# The nano-vllm comparison on one GPU without Slurm, from the repo root, after scripts/lambda_setup.sh:
#   bash scripts/lambda_bench.sh smoke     # 50 conversations, 5 waves, one repeat: check both runners work
#   bash scripts/lambda_bench.sh grid      # 300 conversations, rho 0.6/1/2, 3 repeats
# Finished runs are skipped, so rerunning after an interruption resumes. Run it under tmux or nohup:
# an SSH disconnect would otherwise stop it.
set -euo pipefail
MODEL=~/models/Qwen3-0.6B
TRACE=benchmarks/data/trace_qwen3.json
OURS=~/venvs/tinyserve/bin/python
NANO=~/venvs/nano/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

case "${1:-}" in
    smoke) common=(--conversations 50 --max-waves 5 --rhos 1 --repeats 1 --tag smoke --out-dir benchmarks/results/qwen_smoke) ;;
    # rho 0.6, not 0.5: 300 Qwen conversations at 0.5 need ~72 GiB of KV, more than an 80 GB card leaves.
    grid)  common=(--conversations 300 --rhos 0.6,1,2 --repeats 3 --out-dir benchmarks/results/qwen) ;;
    *) echo "usage: bash scripts/lambda_bench.sh smoke|grid"; exit 1 ;;
esac

"$OURS" -m benchmarks.driver --engine nano --python "$NANO" --model "$MODEL" --trace "$TRACE" "${common[@]}"
"$OURS" -m benchmarks.driver --engine ours --python "$OURS" --model "$MODEL" --trace "$TRACE" --caching on,off "${common[@]}"
