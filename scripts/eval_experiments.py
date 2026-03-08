#!/usr/bin/env python3
"""Evaluate all experiments on fold 0 and produce comparison table."""
from __future__ import annotations
import json, sys
from collections import defaultdict
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

def kfold_val(dataset, fold=0, num_folds=5, seed=SEED):
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
    for l in LABELS:
        if l in text.upper(): return l
    return None

def evaluate(model, tok, val_ds, mode):
    device = model.device
    golds, preds = [], []
    for i, ex in enumerate(val_ds):
        p = build_prompt(ex["claim_text"], ex["caption"], mode)
        inputs = tok(f"[INST] {p} [/INST]", return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attn = inputs.get("attention_mask")
        if attn is not None: attn = attn.to(device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attn,
                                 max_new_tokens=128 if mode == "D" else 256,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        pred = parse_decision(text)
        golds.append(ex["label_3class"])
        preds.append(pred if pred else "NEUTRAL")
        if (i+1) % 50 == 0:
            correct = sum(1 for g, p in zip(golds, preds) if g == p)
            print(f"  [{i+1}/{len(val_ds)}] acc={correct/len(golds):.3f}", flush=True)

    # Compute metrics
    results = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        prec = tp/(tp+fp) if (tp+fp) else 0; rec = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        results[label] = {"P": round(prec,4), "R": round(rec,4), "F1": round(f1,4)}
    results["macro_f1"] = round(float(np.mean([results[l]["F1"] for l in LABELS])), 4)
    cm = defaultdict(lambda: defaultdict(int))
    for g, p in zip(golds, preds): cm[g][p] += 1
    results["confusion"] = {g: dict(cm[g]) for g in LABELS}
    return results

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--exps", nargs="+", default=["A", "B", "C", "D", "E"])
    a = p.parse_args()

    ds = load_dataset(DATASET_ID, split="test")
    val_ds = kfold_val(ds)
    print(f"Val size: {len(val_ds)}", flush=True)

    # Map experiment to (adapter_path, eval_mode)
    exp_info = {
        "A": ("lora_exps/exp_a/adapter", "R"),
        "B": ("lora_exps/exp_b/adapter", "D"),
        "C": ("lora_exps/exp_c/adapter", "D"),
        "D": ("lora_exps/exp_d/adapter", "R"),
        "E": ("lora_exps/exp_e/adapter", "R"),
    }

    all_results = {}
    for exp_name in a.exps:
        adapter_path, mode = exp_info[exp_name]
        if not Path(adapter_path).exists():
            print(f"\n[SKIP] Exp {exp_name}: adapter not found at {adapter_path}", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[EVAL] Experiment {exp_name} (mode={mode})", flush=True)

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        ng = torch.cuda.device_count()
        mm = {i: "32GiB" for i in range(ng)}
        mm["cpu"] = "80GiB"
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                     device_map="auto", max_memory=mm, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token

        result = evaluate(model, tok, val_ds, mode)
        all_results[exp_name] = result

        print(f"\n[RESULT] Exp {exp_name}: Macro-F1={result['macro_f1']}", flush=True)
        for l in LABELS:
            print(f"  {l}: P={result[l]['P']} R={result[l]['R']} F1={result[l]['F1']}", flush=True)
        print(f"  Confusion: {result['confusion']}", flush=True)

        del model, base; torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}", flush=True)
    print("COMPARISON TABLE (Fold 0 val set)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Exp':<6} {'Macro-F1':<10} {'SUP-F1':<10} {'NEU-F1':<10} {'CON-F1':<10}", flush=True)
    print("-" * 46, flush=True)
    # Add baseline
    print(f"{'base':<6} {'0.2519':<10} {'0.1658':<10} {'0.5177':<10} {'0.0721':<10}  (zero-shot D)", flush=True)
    print(f"{'baseR':<6} {'0.3409':<10} {'0.3033':<10} {'0.5443':<10} {'0.1751':<10}  (zero-shot R)", flush=True)
    for exp_name in sorted(all_results):
        r = all_results[exp_name]
        print(f"{exp_name:<6} {r['macro_f1']:<10} {r['SUPPORT']['F1']:<10} {r['NEUTRAL']['F1']:<10} {r['CONTRADICT']['F1']:<10}", flush=True)

    Path("lora_exps/comparison.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to lora_exps/comparison.json", flush=True)

if __name__ == "__main__":
    main()
