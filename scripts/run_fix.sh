#!/bin/bash
set -Eeuo pipefail
export PATH=/home/zechuan/anaconda3/bin:$PATH
source activate musciclaim
cd /home/zechuan/MuSciClaim
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[FIX] Starting at $(date)"

# 1. Train DPO (Exp E)
echo ">>> Training DPO at $(date)"
rm -rf lora_exps/exp_e
python scripts/train_dpo.py
echo ">>> DPO done at $(date)"

# 2. Evaluate C, D, E
echo ">>> Evaluating Exp C (D mode) at $(date)"
python scripts/qlora_eval.py --fold 0 --num-folds 5 --mode D --adapter-dir lora_exps/exp_c/adapter --output-dir lora_exps
echo ">>> Exp C eval done at $(date)"

echo ">>> Evaluating Exp D (R mode) at $(date)"
python scripts/qlora_eval.py --fold 0 --num-folds 5 --mode R --adapter-dir lora_exps/exp_d/adapter --output-dir lora_exps
echo ">>> Exp D eval done at $(date)"

echo ">>> Evaluating Exp E (R mode) at $(date)"
python scripts/qlora_eval.py --fold 0 --num-folds 5 --mode R --adapter-dir lora_exps/exp_e/adapter --output-dir lora_exps
echo ">>> Exp E eval done at $(date)"

echo "[FIX] All done at $(date)"
