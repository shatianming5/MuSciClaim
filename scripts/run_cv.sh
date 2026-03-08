#!/bin/bash
set -Eeuo pipefail
export PATH=/home/zechuan/anaconda3/bin:$PATH
source activate musciclaim
cd /home/zechuan/MuSciClaim
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

MODE="D"
RANK=16
ALPHA=32
NUM_FOLDS=5
EPOCHS=3

echo "[CV] Starting 5-fold QLoRA training, mode=${MODE}, rank=${RANK}"
echo "[CV] Time: $(date)"

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "============================================"
    echo "[CV] Training fold ${FOLD}/${NUM_FOLDS}"
    echo "[CV] Time: $(date)"
    echo "============================================"

    python scripts/qlora_train.py         --fold ${FOLD}         --num-folds ${NUM_FOLDS}         --mode ${MODE}         --epochs ${EPOCHS}         --rank ${RANK}         --alpha ${ALPHA}         --lr 1e-4         --batch-size 4         --grad-accum 4         --output-dir lora_runs
done

echo ""
echo "============================================"
echo "[CV] All folds trained. Starting evaluation..."
echo "[CV] Time: $(date)"
echo "============================================"

python scripts/qlora_eval.py     --all-folds     --num-folds ${NUM_FOLDS}     --mode ${MODE}     --rank ${RANK}     --alpha ${ALPHA}     --output-dir lora_runs

echo ""
echo "[CV] All done! Time: $(date)"
