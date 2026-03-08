#!/usr/bin/env python3
"""Evaluate a LoRA-finetuned Mixtral on MuSciClaims per-fold."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DATASET_ID = "StonyBrookNLP/MuSciClaims"
SEED = 42
LABELS = ["SUPPORT", "NEUTRAL", "CONTRADICT"]

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

def kfold_split(dataset, num_folds, fold, seed=SEED):
    base_ids = sorted(set(dataset["base_claim_id"]))
    rng = np.random.RandomState(seed)
    rng.shuffle(base_ids)
    fs = len(base_ids) // num_folds
    vs, ve = fold * fs, (fold * fs + fs if fold < num_folds - 1 else len(base_ids))
    val_ids = set(base_ids[vs:ve])
    return dataset.select([i for i, b in enumerate(dataset["base_claim_id"]) if b in val_ids])

def parse_decision(text):
    text = text.strip()
    try:
        obj = json.loads(text)
        d = obj.get("decision", "").upper().strip()
        if d in LABELS: return d
    except: pass
    for label in LABELS:
        if label in text.upper(): return label
    return None

def compute_metrics(golds, preds):
    results = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    results["macro_f1"] = round(float(np.mean([results[l]["f1"] for l in LABELS])), 4)
    cm = defaultdict(lambda: defaultdict(int))
    for g, p in zip(golds, preds): cm[g][p] += 1
    results["confusion"] = {g: dict(cm[g]) for g in LABELS}
    results["invalid_count"] = sum(1 for p in preds if p is None)
    results["total"] = len(golds)
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--mode", default="D", choices=["D", "R"])
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--output-dir", default="lora_runs")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)

    # OOM/speed controls
    p.add_argument("--max-length", type=int, default=2048, help="tokenizer truncation length")
    p.add_argument("--max-new-tokens", type=int, default=None, help="override max_new_tokens")
    p.add_argument("--max-samples", type=int, default=0, help="0=all; otherwise limit eval samples")
    p.add_argument("--use-chat-template", action="store_true", help="use tokenizer.apply_chat_template instead of manual [INST]")

    a = p.parse_args()

    ds_full = load_dataset(DATASET_ID, split="test")
    val_ds = kfold_split(ds_full, a.num_folds, a.fold)
    if a.max_samples and a.max_samples > 0:
        val_ds = val_ds.select(list(range(min(len(val_ds), a.max_samples))))
    print(f"[INFO] Fold {a.fold}, Val size: {len(val_ds)}", flush=True)

    adapter_path = a.adapter_dir or str(Path(a.output_dir) / f"fold{a.fold}_{a.mode}_r{a.rank}_a{a.alpha}" / "adapter")
    print(f"[INFO] Loading adapter: {adapter_path}", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    ng = torch.cuda.device_count()
    mm = {i: "32GiB" for i in range(ng)}
    mm["cpu"] = "80GiB"

    print(f"[INFO] Loading base model 4-bit across {ng} GPUs...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                  device_map="auto", max_memory=mm, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = model.device
    golds, preds = [], []

    for i, ex in enumerate(val_ds):
        prompt = build_prompt_text(ex["claim_text"], ex["caption"], a.mode)

        if a.use_chat_template:
            msgs = [
                {
                    "role": "system",
                    "content": "You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning.",
                },
                {"role": "user", "content": prompt},
            ]
            full_prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            full_prompt = f"[INST] {prompt} [/INST]"

        inputs = tok(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=a.max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)

        max_new = a.max_new_tokens
        if max_new is None:
            max_new = 128 if a.mode == "D" else 256

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
        pred = parse_decision(text)
        golds.append(ex["label_3class"])
        preds.append(pred if pred else "NEUTRAL")

        # reduce fragmentation
        if (i + 1) % 20 == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        if (i + 1) % 50 == 0:
            interim = compute_metrics(golds, preds)
            print(f"  [{i+1}/{len(val_ds)}] interim Macro-F1={interim['macro_f1']:.4f}", flush=True)

    result = compute_metrics(golds, preds)
    print(f"\n[RESULT] Fold {a.fold}: Macro-F1={result['macro_f1']:.4f}", flush=True)
    for l in LABELS:
        print(f"  {l}: P={result[l]['precision']:.4f} R={result[l]['recall']:.4f} F1={result[l]['f1']:.4f}", flush=True)
    print(f"  Confusion: {result['confusion']}", flush=True)
    print(f"  Invalid: {result['invalid_count']}/{result['total']}", flush=True)

    out_path = Path(a.output_dir) / f"fold{a.fold}_{a.mode}_r{a.rank}_a{a.alpha}" / "eval_result.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[INFO] Result saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
