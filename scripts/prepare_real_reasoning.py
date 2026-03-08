#!/usr/bin/env python3
"""Extract correct R-mode predictions as training data with real reasoning chains."""
import json
from pathlib import Path

pred_file = Path("runs/full_mixtral_8x7b/ours/c_only/R/predictions.jsonl")
out_file = Path("scripts/real_reasoning_data.jsonl")

records = []
with pred_file.open() as f:
    for line in f:
        r = json.loads(line)
        if r["label_gold"] == r["label_pred"] and r["raw_text"] and not r["invalid_output"]:
            records.append({
                "claim_id": r["claim_id"],
                "base_claim_id": r["base_claim_id"],
                "label": r["label_gold"],
                "raw_text": r["raw_text"],  # The actual model output with reasoning
            })

with out_file.open("w") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

print(f"Extracted {len(records)} correct R-mode predictions")

# Also extract wrong predictions for DPO
wrong_file = Path("scripts/wrong_predictions.jsonl")
wrong = []
with pred_file.open() as f:
    for line in f:
        r = json.loads(line)
        if r["label_gold"] != r["label_pred"] and r["raw_text"] and not r["invalid_output"]:
            wrong.append({
                "claim_id": r["claim_id"],
                "base_claim_id": r["base_claim_id"],
                "label_gold": r["label_gold"],
                "label_pred": r["label_pred"],
                "raw_text": r["raw_text"],
            })

with wrong_file.open("w") as f:
    for rec in wrong:
        f.write(json.dumps(rec) + "\n")
print(f"Extracted {len(wrong)} wrong R-mode predictions for DPO")
