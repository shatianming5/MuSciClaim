#!/usr/bin/env python3
"""QLoRA fine-tuning for Mixtral-8x7B on MuSciClaims with K-fold CV."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DATASET_ID = "StonyBrookNLP/MuSciClaims"
MAX_SEQ_LEN = 512
SEED = 42

def build_prompt_text(claim, caption, mode):
    header = ("You are an AI model tasked with verifying claims related to visual evidence "
              "using zero-shot learning.\nYour job is to analyze the given figure(s) and caption(s) "
              "to decide whether they SUPPORT or CONTRADICT or are NEUTRAL with respect to the claim.\n")
    body = f"CLAIM: {claim}\nFIGURE: (not provided)\nIMAGE CAPTION(S): {caption}\n"
    if mode == "D":
        schema = ("After completing your analysis, output exactly one JSON object with exactly one key: "
                  '"decision".\nFor "decision", output exactly one word: '
                  '"SUPPORT" or "CONTRADICT" or "NEUTRAL" (uppercase).\n'
                  "The output must start with { and end with }.\n")
    else:
        schema = ("After completing your analysis, output exactly one JSON object with exactly two keys: "
                  '"reasoning" and "decision".\n- "reasoning": 1-2 sentences grounded in the figure/caption '
                  "(mention axes/legend/trends/panels when relevant).\n"
                  '- "decision": "SUPPORT" or "CONTRADICT" or "NEUTRAL" (uppercase).\nNo extra text.\n')
    return header + body + schema

def build_target(label, mode):
    if mode == "D":
        return json.dumps({"decision": label})
    return json.dumps({"reasoning": "Based on the caption evidence.", "decision": label})

def kfold_split(dataset, num_folds, fold, seed=SEED):
    base_ids = sorted(set(dataset["base_claim_id"]))
    rng = np.random.RandomState(seed)
    rng.shuffle(base_ids)
    fs = len(base_ids) // num_folds
    vs, ve = fold * fs, (fold * fs + fs if fold < num_folds - 1 else len(base_ids))
    val_ids = set(base_ids[vs:ve])
    tr = [i for i, b in enumerate(dataset["base_claim_id"]) if b not in val_ids]
    va = [i for i, b in enumerate(dataset["base_claim_id"]) if b in val_ids]
    return dataset.select(tr), dataset.select(va)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--mode", default="D", choices=["D", "R"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--output-dir", default="lora_runs")
    a = p.parse_args()

    run_name = f"fold{a.fold}_{a.mode}_r{a.rank}_a{a.alpha}"
    out = Path(a.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Fold {a.fold}/{a.num_folds}, mode={a.mode}, rank={a.rank}")

    ds = load_dataset(DATASET_ID, split="test")
    train_ds, val_ds = kfold_split(ds, a.num_folds, a.fold)
    print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}")
    from collections import Counter
    print(f"[INFO] Train labels: {dict(Counter(train_ds['label_3class']))}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    def fmt(ex):
        pr = build_prompt_text(ex["claim_text"], ex["caption"], a.mode)
        tgt = build_target(ex["label_3class"], a.mode)
        return {"text": f"[INST] {pr} [/INST] {tgt}"}

    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(fmt, remove_columns=val_ds.column_names)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    ng = torch.cuda.device_count()
    mm = {i: "32GiB" for i in range(ng)}
    mm["cpu"] = "80GiB"
    print(f"[INFO] Loading 4-bit model across {ng} GPUs...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                  device_map="auto", max_memory=mm, torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, target_modules=["q_proj", "v_proj"],
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    cfg = SFTConfig(output_dir=str(out), max_length=MAX_SEQ_LEN, num_train_epochs=a.epochs,
                    per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
                    learning_rate=a.lr, warmup_ratio=0.1, weight_decay=0.01, bf16=True,
                    logging_steps=10, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                    report_to="none", seed=SEED, dataloader_num_workers=2,
                    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                    optim="paged_adamw_8bit")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok)
    print("[INFO] Starting training...")
    trainer.train()

    ap = out / "adapter"
    model.save_pretrained(str(ap))
    tok.save_pretrained(str(ap))
    print(f"[INFO] LoRA adapter saved to {ap}")
    (out / "fold_info.json").write_text(json.dumps({"fold": a.fold, "num_folds": a.num_folds,
        "mode": a.mode, "rank": a.rank, "alpha": a.alpha,
        "train_size": len(train_ds), "val_size": len(val_ds)}, indent=2))
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
