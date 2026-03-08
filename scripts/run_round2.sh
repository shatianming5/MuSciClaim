#!/bin/bash
set -Eeuo pipefail
source /home/zechuan/anaconda3/etc/profile.d/conda.sh
conda activate musciclaim
cd /home/zechuan/MuSciClaim
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

echo "[ROUND2] start $(date)"

# Round-1: clean prompt + clean rejected + longer context
rm -rf lora_exps/exp_e_v2
python scripts/train_dpo_v2.py --out lora_exps/exp_e_v2 --max-length 2048 --beta 0.1 --lr 5e-6 --epochs 1

# Eval C/D/E_v2 sequentially to reduce OOM
for name in exp_c exp_d exp_e_v2; do
  echo "[EVAL] ${name} $(date)"
  pkill -f qlora_eval.py || true
  python - <<'PY'
import torch,gc
try:
  torch.cuda.empty_cache()
  gc.collect()
except Exception:
  pass
PY

  if [ "$name" = "exp_c" ]; then MODE="D"; else MODE="R"; fi
  ADAPTER="lora_exps/${name}/adapter"
  python scripts/qlora_eval.py --fold 0 --num-folds 5 --mode ${MODE} --adapter-dir ${ADAPTER} --output-dir lora_exps
  echo "[EVAL_DONE] ${name} $(date)"
done

echo "[ROUND2] done $(date)"
