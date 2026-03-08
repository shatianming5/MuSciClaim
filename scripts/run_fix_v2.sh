#!/bin/bash
set -Eeuo pipefail
source /home/zechuan/anaconda3/etc/profile.d/conda.sh
conda activate musciclaim
cd /home/zechuan/MuSciClaim
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use all GPUs (trainer uses device_map=auto)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

log=fix_v2.log

echo "[FIX_V2] Starting $(date)" | tee -a $log

# 1) Train DPO E_v2 (clean prompt + clean rejected + longer context)
rm -rf lora_exps/exp_e_v2
python scripts/train_dpo_v2.py --out lora_exps/exp_e_v2 --max-length 2048 --beta 0.1 --lr 5e-6 --epochs 1 2>&1 | tee -a $log

echo "[FIX_V2] DPO done $(date)" | tee -a $log

# 2) Eval C/D/E sequentially (safe params)
for name in exp_c exp_d exp_e_v2; do
  echo "[FIX_V2] Evaluating ${name} $(date)" | tee -a $log
  pkill -f qlora_eval.py || true
  python - <<'PY'
import gc
import torch
try:
  torch.cuda.empty_cache()
  gc.collect()
except Exception:
  pass
PY

  if [ "$name" = "exp_c" ]; then MODE="D"; else MODE="R"; fi
  ADAPTER="lora_exps/${name}/adapter"

  python scripts/qlora_eval.py \
    --fold 0 --num-folds 5 --mode ${MODE} \
    --adapter-dir ${ADAPTER} --output-dir lora_exps \
    --max-length 2048 --max-new-tokens 192 --use-chat-template \
    2>&1 | tee -a $log

  echo "[FIX_V2] Eval ${name} done $(date)" | tee -a $log

done

echo "[FIX_V2] All done $(date)" | tee -a $log
