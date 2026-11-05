#!/bin/bash
# The nano-vllm comparison on a rented Linux GPU machine without Slurm, from the repo root, after
# scripts/lambda_setup.sh:
#   bash scripts/lambda_bench.sh smoke     # 50 conversations, 5 waves, one repeat: check both runners work
#   bash scripts/lambda_bench.sh grid      # 300 conversations, rho 0.6/1/2, 3 repeats
# Finished runs are skipped, so rerunning after an interruption resumes. Run it under tmux or nohup:
# an SSH disconnect would otherwise stop it.
#
# On one GPU everything runs in turn. On several, nano-vllm gets GPU 0 and our runs are split by rho
# over the rest, each GPU running one process at a time. Only one nano-vllm process can run at once:
# it always opens its process group on port 2333, even on a single GPU. Per-GPU logs go to logs/.
set -euo pipefail
MODEL=~/models/Qwen3-0.6B
TRACE=benchmarks/data/trace_qwen3.json
OURS=~/venvs/tinyserve/bin/python
NANO=~/venvs/nano/bin/python
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NGPU=$(nvidia-smi -L | wc -l)

MODE=${1:-}
case "$MODE" in
    smoke) RHOS=1; common=(--conversations 50 --max-waves 5 --repeats 1 --tag smoke --out-dir benchmarks/results/qwen_smoke) ;;
    # rho 0.6, not 0.5: 300 Qwen conversations at 0.5 need ~72 GiB of KV, more than an 80 GB card leaves.
    grid)  RHOS=0.6,1,2; common=(--conversations 300 --repeats 3 --out-dir benchmarks/results/qwen) ;;
    *) echo "usage: bash scripts/lambda_bench.sh smoke|grid"; exit 1 ;;
esac
mkdir -p logs

nano() { "$OURS" -m benchmarks.driver --engine nano --python "$NANO" --model "$MODEL" --trace "$TRACE" --rhos "$1" "${common[@]}"; }
ours() { "$OURS" -m benchmarks.driver --engine ours --python "$OURS" --model "$MODEL" --trace "$TRACE" --rhos "$1" --caching on,off "${common[@]}"; }

if (( NGPU == 1 )); then
    nano "$RHOS"
    ours "$RHOS"
    exit
fi

IFS=, read -ra rho_list <<< "$RHOS"
groups=()
for i in "${!rho_list[@]}"; do
    g=$(( i % (NGPU - 1) ))
    groups[g]="${groups[g]:+${groups[g]},}${rho_list[i]}"
done

pids=()
CUDA_VISIBLE_DEVICES=0 nano "$RHOS" > "logs/${MODE}_nano_gpu0.log" 2>&1 &
pids+=($!)
echo "GPU 0: nano-vllm, rho $RHOS -> logs/${MODE}_nano_gpu0.log"
for g in "${!groups[@]}"; do
    gpu=$(( g + 1 ))
    CUDA_VISIBLE_DEVICES=$gpu ours "${groups[g]}" > "logs/${MODE}_ours_gpu${gpu}.log" 2>&1 &
    pids+=($!)
    echo "GPU $gpu: ours, rho ${groups[g]} -> logs/${MODE}_ours_gpu${gpu}.log"
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
(( status == 0 )) && echo "all runs finished" || echo "some runs failed; see logs/${MODE}_*.log"
exit $status
