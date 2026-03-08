#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

MODEL_ID_DEFAULT = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DATASET_ID_DEFAULT = "StonyBrookNLP/MuSciClaims"
LABELS = {"SUPPORT", "CONTRADICT", "NEUTRAL"}


def build_messages(claim: str, caption: str):
    system = (
        "You are an AI model tasked with verifying claims related to visual evidence using zero-shot learning. "
        "Decide whether the caption SUPPORTS or CONTRADICTS or is NEUTRAL with respect to the claim."
    )
    user = (
        f"CLAIM: {claim}\n"
        f"FIGURE: (not provided)\n"
        f"IMAGE CAPTION(S): {caption}\n\n"
        "After completing your analysis, output exactly one JSON object with exactly two keys: "
        "\"reasoning\" and \"decision\".\n"
        "- \"reasoning\": 1-2 sentences grounded in the caption.\n"
        "- \"decision\": \"SUPPORT\" or \"CONTRADICT\" or \"NEUTRAL\" (uppercase).\n"
        "No extra text."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_json_answer(raw_text: str, *, fallback_decision: str) -> str:
    """Return a strict JSON string with keys reasoning/decision."""
    decision = fallback_decision
    reasoning = "The caption does not provide clear evidence about the claim."

    def _try_parse(s: str):
        nonlocal decision, reasoning
        j = json.loads(s)
        if isinstance(j, dict):
            d = str(j.get("decision", "")).strip().upper()
            r = str(j.get("reasoning", "")).strip()
            if d in LABELS:
                decision = d
            if r:
                reasoning = r

    try:
        _try_parse(raw_text)
    except Exception:
        s = raw_text
        i, k = s.find("{"), s.rfind("}")
        if i != -1 and k != -1 and k > i:
            try:
                _try_parse(s[i : k + 1])
            except Exception:
                pass

    return json.dumps({"reasoning": reasoning, "decision": decision}, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_DEFAULT)
    ap.add_argument("--dataset", default=DATASET_ID_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--num-folds", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", default="lora_exps/exp_e_v2")
    ap.add_argument("--pairs-file", default="scripts/wrong_predictions.jsonl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = load_dataset(args.dataset, split="test")
    base_ids = sorted(set(ds["base_claim_id"]))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(base_ids)
    fs = len(base_ids) // args.num_folds
    start = args.fold * fs
    end = (args.fold + 1) * fs if args.fold < args.num_folds - 1 else len(base_ids)
    val_ids = set(base_ids[start:end])

    wrong = {}
    with open(args.pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            wrong[r["claim_id"]] = r

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dpo_train, dpo_val = [], []
    for ex in ds:
        cid = ex["claim_id"]
        bid = ex["base_claim_id"]
        if cid not in wrong:
            continue
        is_val = bid in val_ids
        w = wrong[cid]

        prompt_text = tok.apply_chat_template(
            build_messages(ex["claim_text"], ex["caption"]),
            tokenize=False,
            add_generation_prompt=True,
        )

        gold = str(ex["label_3class"]).strip().upper()
        if gold not in LABELS:
            continue

        chosen = normalize_json_answer(
            json.dumps({"reasoning": "", "decision": gold}),
            fallback_decision=gold,
        )

        rej = normalize_json_answer(str(w.get("raw_text", "")), fallback_decision="NEUTRAL")
        try:
            rej_obj = json.loads(rej)
        except Exception:
            rej_obj = {"reasoning": "", "decision": "NEUTRAL"}
        bad = str(rej_obj.get("decision", "NEUTRAL")).strip().upper()
        if bad == gold or bad not in LABELS:
            bad = {"SUPPORT": "NEUTRAL", "CONTRADICT": "NEUTRAL", "NEUTRAL": "SUPPORT"}.get(gold, "NEUTRAL")
        rejected = json.dumps(
            {
                "reasoning": str(rej_obj.get("reasoning", "The caption does not provide clear evidence."))[:400],
                "decision": bad,
            },
            ensure_ascii=False,
        )

        pair = {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}
        (dpo_val if is_val else dpo_train).append(pair)

    print(f"DPO pairs: train={len(dpo_train)}, val={len(dpo_val)}", flush=True)
    train_ds = Dataset.from_list(dpo_train)
    val_ds = Dataset.from_list(dpo_val)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    ng = torch.cuda.device_count()
    mm = {i: "32GiB" for i in range(ng)}
    mm["cpu"] = "80GiB"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        max_memory=mm,
        torch_dtype=torch.bfloat16,
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dpo_cfg = DPOConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_length=args.max_length,
        beta=args.beta,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=lora_cfg,
    )

    print("Starting DPO training...", flush=True)
    trainer.train()

    apath = out / "adapter"
    trainer.model.save_pretrained(str(apath))
    tok.save_pretrained(str(apath))
    print(f"DPO adapter saved to {apath}", flush=True)


if __name__ == "__main__":
    main()
