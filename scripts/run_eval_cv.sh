#!/bin/bash
set -Eeuo pipefail
export PATH=/home/zechuan/anaconda3/bin:$PATH
source activate musciclaim
cd /home/zechuan/MuSciClaim
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTHONUNBUFFERED=1

MODE="D"
RANK=16
ALPHA=32

echo "[EVAL] Starting per-fold evaluation"
echo "[EVAL] Time: $(date)"

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "=== Evaluating fold ${FOLD} ==="
    echo "Time: $(date)"
    python scripts/qlora_eval.py --fold ${FOLD} --num-folds 5 --mode ${MODE} --rank ${RANK} --alpha ${ALPHA} --output-dir lora_runs
done

echo ""
echo "[EVAL] All folds done! Time: $(date)"
