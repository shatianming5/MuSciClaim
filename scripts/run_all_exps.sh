#!/bin/bash
set -Eeuo pipefail
export PATH=/home/zechuan/anaconda3/bin:$PATH
source activate musciclaim
cd /home/zechuan/MuSciClaim
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTHONUNBUFFERED=1

echo "[ALL] Starting all experiments at $(date)"
rm -rf lora_exps

echo ""
echo "########## TRAINING PHASE ##########"
for EXP in A B C D E; do
    echo ""
    echo ">>> Training Exp ${EXP} at $(date)"
    python scripts/train_experiments.py --exp ${EXP} --fold 0 2>&1 || echo "[WARN] Exp ${EXP} training failed"
    echo ">>> Exp ${EXP} training done at $(date)"
done

echo ""
echo "########## EVALUATION PHASE ##########"
echo "Starting evaluation at $(date)"
python scripts/eval_experiments.py 2>&1

echo ""
echo "[ALL] Everything done at $(date)"
