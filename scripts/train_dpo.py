#!/usr/bin/env python3
import json, os
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DATASET_ID = "StonyBrookNLP/MuSciClaims"
SEED = 42

def build_prompt(claim, caption):
    header = ("You are an AI model tasked with verifying claims related to visual evidence "
              "using zero-shot learning.\nYour job is to analyze the given figure(s) and caption(s) "
              "to decide whether they SUPPORT or CONTRADICT or are NEUTRAL with respect to the claim.\n")
    body = f"CLAIM: {claim}\nFIGURE: (not provided)\nIMAGE CAPTION(S): {caption}\n"
    schema = ("After completing your analysis, output exactly one JSON object with exactly two keys: "
              '"reasoning" and "decision".\n- "reasoning": 1-2 sentences grounded in the figure/caption '
              "(mention axes/legend/trends/panels when relevant).\n"
              '- "decision": "SUPPORT" or "CONTRADICT" or "NEUTRAL" (uppercase).\nNo extra text.\n')
    return header + body + schema

ds = load_dataset(DATASET_ID, split="test")
base_ids = sorted(set(ds["base_claim_id"]))
rng = np.random.RandomState(SEED)
rng.shuffle(base_ids)
fs = len(base_ids) // 5
val_ids = set(base_ids[:fs])

wrong = {}
with open("scripts/wrong_predictions.jsonl") as f:
    for line in f:
        r = json.loads(line)
        wrong[r["claim_id"]] = r

dpo_train, dpo_val = [], []
for ex in ds:
    cid = ex["claim_id"]
    bid = ex["base_claim_id"]
    is_val = bid in val_ids
    if cid not in wrong:
        continue
    w = wrong[cid]
    p = build_prompt(ex["claim_text"], ex["caption"])
    prompt_text = f"[INST] {p} [/INST]"
    chosen = json.dumps({"reasoning": "The caption provides relevant evidence for this claim.", "decision": ex["label_3class"]})
    rejected = w["raw_text"]
    pair = {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}
    if is_val:
        dpo_val.append(pair)
    else:
        dpo_train.append(pair)

print(f"DPO pairs: train={len(dpo_train)}, val={len(dpo_val)}", flush=True)
train_ds = Dataset.from_list(dpo_train)
val_ds = Dataset.from_list(dpo_val)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
ng = torch.cuda.device_count()
mm = {i: "32GiB" for i in range(ng)}
mm["cpu"] = "80GiB"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                              device_map="auto", max_memory=mm, torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
out = Path("lora_exps/exp_e")
out.mkdir(parents=True, exist_ok=True)

dpo_cfg = DPOConfig(
    output_dir=str(out), num_train_epochs=1,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=5e-6, warmup_ratio=0.1, bf16=True,
    logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=1, report_to="none", seed=SEED,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit", max_length=512,
)
trainer = DPOTrainer(model=model, args=dpo_cfg, train_dataset=train_ds, eval_dataset=val_ds,
                     processing_class=tok, peft_config=lora_cfg)
print("Starting DPO training...", flush=True)
trainer.train()
ap = out / "adapter"
trainer.model.save_pretrained(str(ap))
tok.save_pretrained(str(ap))
print(f"DPO adapter saved to {ap}", flush=True)
