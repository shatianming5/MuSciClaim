#!/bin/bash
set -Eeuo pipefail
export PATH=/home/zechuan/anaconda3/bin:$PATH
source activate musciclaim
cd /home/zechuan/MuSciClaim
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
musciclaim-eval --run-config configs/run.yaml --models-config configs/models.yaml --run-id full_mixtral_8x7b
