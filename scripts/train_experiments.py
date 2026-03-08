#!/usr/bin/env python3
"""Unified QLoRA experiment runner. Trains fold 0 for quick comparison.

Experiments:
  A: R-mode template targets, lr=1e-4, epoch=3
  B: D-mode, lr=2e-5, epoch=1 (gentle tuning)
  C: D-mode, completion-only loss, lr=1e-4, epoch=3
  D: Real reasoning chains as targets, lr=2e-5, epoch=1
  E: DPO preference learning
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DATASET_ID = "StonyBrookNLP/MuSciClaims"
SEED = 42

def build_prompt(claim, caption, mode):
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

def kfold_split(dataset, num_folds, fold, seed=SEED):
    base_ids = sorted(set(dataset["base_claim_id"]))
    rng = np.random.RandomState(seed)
    rng.shuffle(base_ids)
    fs = len(base_ids) // num_folds
    vs, ve = fold * fs, (fold * fs + fs if fold < num_folds - 1 else len(base_ids))
    val_ids = set(base_ids[vs:ve])
    tr = [i for i, b in enumerate(dataset["base_claim_id"]) if b not in val_ids]
    va = [i for i, b in enumerate(dataset["base_claim_id"]) if b in val_ids]
    return dataset.select(tr), dataset.select(va), val_ids

def load_model_and_lora(rank=16, alpha=32):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    ng = torch.cuda.device_count()
    mm = {i: "32GiB" for i in range(ng)}
    mm["cpu"] = "80GiB"
    print(f"[INFO] Loading 4-bit model across {ng} GPUs...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                  device_map="auto", max_memory=mm, torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=["q_proj", "v_proj"],
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return model, tok

def train_sft(model, tok, train_ds, val_ds, out_dir, lr, epochs, batch_size=2, grad_accum=8, completion_only=False):
    cfg = SFTConfig(
        output_dir=str(out_dir), max_length=512, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01, bf16=True,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=SEED, dataloader_num_workers=2,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok)
    print(f"[INFO] Training: lr={lr}, epochs={epochs}, completion_only={completion_only}", flush=True)
    trainer.train()
    ap = out_dir / "adapter"
    model.save_pretrained(str(ap))
    tok.save_pretrained(str(ap))
    print(f"[INFO] Adapter saved to {ap}", flush=True)
    return ap

def run_exp_a(ds, fold=0):
    """Exp A: R-mode template, lr=1e-4, epoch=3"""
    print("\n" + "="*60, flush=True)
    print("[EXP A] R-mode template targets, lr=1e-4, epoch=3", flush=True)
    train_ds, val_ds, _ = kfold_split(ds, 5, fold)
    def fmt(ex):
        p = build_prompt(ex["claim_text"], ex["caption"], "R")
        t = json.dumps({"reasoning": "Based on the caption evidence.", "decision": ex["label_3class"]})
        return {"text": f"[INST] {p} [/INST] {t}"}
    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(fmt, remove_columns=val_ds.column_names)
    model, tok = load_model_and_lora()
    out = Path("lora_exps/exp_a")
    out.mkdir(parents=True, exist_ok=True)
    train_sft(model, tok, train_ds, val_ds, out, lr=1e-4, epochs=3)
    del model; torch.cuda.empty_cache()

def run_exp_b(ds, fold=0):
    """Exp B: D-mode, lr=2e-5, epoch=1"""
    print("\n" + "="*60, flush=True)
    print("[EXP B] D-mode gentle tuning, lr=2e-5, epoch=1", flush=True)
    train_ds, val_ds, _ = kfold_split(ds, 5, fold)
    def fmt(ex):
        p = build_prompt(ex["claim_text"], ex["caption"], "D")
        t = json.dumps({"decision": ex["label_3class"]})
        return {"text": f"[INST] {p} [/INST] {t}"}
    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(fmt, remove_columns=val_ds.column_names)
    model, tok = load_model_and_lora()
    out = Path("lora_exps/exp_b")
    out.mkdir(parents=True, exist_ok=True)
    train_sft(model, tok, train_ds, val_ds, out, lr=2e-5, epochs=1)
    del model; torch.cuda.empty_cache()

def run_exp_c(ds, fold=0):
    """Exp C: D-mode, completion-only loss via chat template formatting"""
    print("\n" + "="*60, flush=True)
    print("[EXP C] D-mode completion-only loss, lr=1e-4, epoch=3", flush=True)
    train_ds, val_ds, _ = kfold_split(ds, 5, fold)
    # Use special formatting: prompt as user, target as assistant
    def fmt(ex):
        p = build_prompt(ex["claim_text"], ex["caption"], "D")
        t = json.dumps({"decision": ex["label_3class"]})
        # Use <s>[INST] prompt [/INST] target </s> with clear boundary
        return {"text": f"<s>[INST] {p} [/INST] {t}</s>"}
    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(fmt, remove_columns=val_ds.column_names)
    model, tok = load_model_and_lora()
    out = Path("lora_exps/exp_c")
    out.mkdir(parents=True, exist_ok=True)
    cfg = SFTConfig(
        output_dir=str(out), max_length=512, num_train_epochs=3,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=1e-4, warmup_ratio=0.1, weight_decay=0.01, bf16=True,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=SEED, dataloader_num_workers=2,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        completion_only_loss=True,
        eos_token="</s>",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok)
    trainer.train()
    ap = out / "adapter"
    model.save_pretrained(str(ap))
    tok.save_pretrained(str(ap))
    del model; torch.cuda.empty_cache()

def run_exp_d(ds, fold=0):
    """Exp D: Real reasoning chains + lr=2e-5, epoch=1"""
    print("\n" + "="*60, flush=True)
    print("[EXP D] Real reasoning chains, lr=2e-5, epoch=1", flush=True)
    _, _, val_ids = kfold_split(ds, 5, fold)

    # Load real reasoning data
    real_data = {}
    with open("scripts/real_reasoning_data.jsonl") as f:
        for line in f:
            r = json.loads(line)
            real_data[r["claim_id"]] = r["raw_text"]

    # For correct predictions: use real model output
    # For wrong predictions: use gold label with template reasoning
    ds_full = load_dataset(DATASET_ID, split="test")
    train_texts = []
    val_texts = []
    for ex in ds_full:
        p = build_prompt(ex["claim_text"], ex["caption"], "R")
        cid = ex["claim_id"]
        bid = ex["base_claim_id"]
        is_val = bid in val_ids

        if cid in real_data:
            # Use the real model output (correct prediction)
            target = real_data[cid]
        else:
            # Use gold label with template
            target = json.dumps({"reasoning": "The caption provides relevant evidence for this claim.",
                                 "decision": ex["label_3class"]})

        text = f"[INST] {p} [/INST] {target}"
        if is_val:
            val_texts.append({"text": text})
        else:
            train_texts.append({"text": text})

    train_ds = Dataset.from_list(train_texts)
    val_ds = Dataset.from_list(val_texts)
    print(f"[INFO] Train: {len(train_ds)} (real reasoning where available), Val: {len(val_ds)}", flush=True)

    model, tok = load_model_and_lora()
    out = Path("lora_exps/exp_d")
    out.mkdir(parents=True, exist_ok=True)
    train_sft(model, tok, train_ds, val_ds, out, lr=2e-5, epochs=1)
    del model; torch.cuda.empty_cache()

def run_exp_e(ds, fold=0):
    """Exp E: DPO preference learning"""
    print("\n" + "="*60, flush=True)
    print("[EXP E] DPO preference learning", flush=True)
    _, _, val_ids = kfold_split(ds, 5, fold)

    # Load correct and wrong predictions
    correct = {}
    with open("scripts/real_reasoning_data.jsonl") as f:
        for line in f:
            r = json.loads(line)
            correct[r["claim_id"]] = r["raw_text"]

    wrong = {}
    with open("scripts/wrong_predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            wrong[r["claim_id"]] = r["raw_text"]

    # Build DPO pairs: need same claim_id with both correct and wrong
    # For claims where model was wrong: chosen=gold template, rejected=wrong output
    ds_full = load_dataset(DATASET_ID, split="test")
    dpo_train = []
    dpo_val = []

    for ex in ds_full:
        cid = ex["claim_id"]
        bid = ex["base_claim_id"]
        is_val = bid in val_ids
        p = build_prompt(ex["claim_text"], ex["caption"], "R")
        prompt_text = f"[INST] {p} [/INST]"

        if cid in wrong:
            # Model was wrong -> construct preference pair
            chosen = json.dumps({"reasoning": "The caption provides relevant evidence for this claim.",
                                 "decision": ex["label_3class"]})
            rejected = wrong[cid]
            pair = {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}
            if is_val:
                dpo_val.append(pair)
            else:
                dpo_train.append(pair)

    print(f"[INFO] DPO pairs: train={len(dpo_train)}, val={len(dpo_val)}", flush=True)

    if len(dpo_train) < 10:
        print("[WARN] Too few DPO pairs, skipping", flush=True)
        return

    from trl import DPOConfig, DPOTrainer
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
        optim="paged_adamw_8bit",
        max_length=512, max_prompt_length=450,
    )
    trainer = DPOTrainer(model=model, args=dpo_cfg, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=lora_cfg)
    trainer.train()
    ap = out / "adapter"
    trainer.model.save_pretrained(str(ap))
    tok.save_pretrained(str(ap))
    del model; torch.cuda.empty_cache()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["A", "B", "C", "D", "E", "all"])
    p.add_argument("--fold", type=int, default=0)
    a = p.parse_args()

    ds = load_dataset(DATASET_ID, split="test")
    exps = [a.exp] if a.exp != "all" else ["A", "B", "C", "D", "E"]

    for exp in exps:
        print(f"\n{'#'*60}", flush=True)
        print(f"# Starting Experiment {exp}", flush=True)
        print(f"{'#'*60}", flush=True)
        if exp == "A": run_exp_a(ds, a.fold)
        elif exp == "B": run_exp_b(ds, a.fold)
        elif exp == "C": run_exp_c(ds, a.fold)
        elif exp == "D": run_exp_d(ds, a.fold)
        elif exp == "E": run_exp_e(ds, a.fold)

if __name__ == "__main__":
    main()
