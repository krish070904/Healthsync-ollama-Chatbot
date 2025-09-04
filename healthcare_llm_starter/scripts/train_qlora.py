#!/usr/bin/env python
"""
Minimal QLoRA SFT recipe for medical QA (MedMCQA/MedQA/PubMedQA).

Run on Colab/Kaggle or a GPU machine. If bitsandbytes fails on Windows, use Linux/WSL or a cloud notebook.

Steps:
1) pip install -r requirements.txt
2) (optional) set HF_TOKEN env if models need auth
3) python scripts/train_qlora.py --base Qwen/Qwen2.5-7B-Instruct --epochs 1
"""

import argparse, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def format_example_mcq(example):
    q = example.get("question", "")
    options = []
    for k in ["opa","opb","opc","opd","choiceA","choiceB","choiceC","choiceD","A","B","C","D"]:
        if k in example and example[k]:
            options.append(example[k])
    if not options and "choices" in example:
        options = example["choices"]

    ans = example.get("cop", None) or example.get("answer", None) or example.get("correct", None)
    if isinstance(ans, int):
        ans_letter = ["A","B","C","D"][ans] if 0 <= ans < 4 else "A"
    elif isinstance(ans, str):
        ans_letter = ans.strip().upper()[0]
    else:
        ans_letter = "A"

    opts_str = "\\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options[:4])])
    prompt = (
        "You are a medical assistant for study and educational purposes.\\n"
        "Read the question and choose ONE best option (A/B/C/D). "
        "Only output this JSON:\\n"
        '{"answer": "X", "reason": "one-sentence justification"}\\n\\n'
        f"Question:\\n{q}\\n\\nOptions:\\n{opts_str}\\n"
    )
    target = json.dumps({"answer": ans_letter, "reason": "Grounded in medical knowledge."})
    return {"text": prompt + target}

def load_medmcqa(split="train"):
    try:
        ds = load_dataset("openlifescienceai/medmcqa", split=split)
    except Exception:
        ds = load_dataset("medmcqa", split=split)
    return ds

def load_medqa(split="train"):
    return load_dataset("GBaker/MedQA-USMLE-4-options", split=split)

def load_pubmedqa(split="train"):
    return load_dataset("qiaojin/PubMedQA", split=split)

def prepare_dataset(sample_size=5000):
    d1 = load_medmcqa("train").shuffle(seed=42).select(range(min(sample_size, 8000)))
    d2 = load_medqa("train").shuffle(seed=42).select(range(min(sample_size, 4000)))
    def fmt_pubmed(x):
        q = x.get("question", "")
        opts = ["Yes","No","Maybe"]
        letter = {"yes":"A","no":"B","maybe":"C"}.get(x.get("final_decision","maybe").strip().lower(), "C")
        opts_str = "\\n".join([f"{chr(65+i)}. {o}" for i,o in enumerate(opts)])
        prompt = (
            "You are a medical assistant for study and educational purposes.\\n"
            "Read the question and choose ONE best option (A/B/C). "
            "Only output this JSON:\\n"
            '{"answer": "X", "reason": "one-sentence justification"}\\n\\n'
            f"Question:\\n{q}\\n\\nOptions:\\n{opts_str}\\n"
        )
        import json as _json
        target = _json.dumps({"answer": letter, "reason": "Grounded in research evidence."})
        return {"text": prompt + target}
    d3 = load_pubmedqa("train").shuffle(seed=42).select(range(min(sample_size//4, 1000))).map(fmt_pubmed, remove_columns=load_pubmedqa("train").column_names)

    d1 = d1.map(format_example_mcq, remove_columns=d1.column_names)
    d2 = d2.map(format_example_mcq, remove_columns=d2.column_names)

    mix = d1.concatenate(d2).concatenate(d3)
    return mix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--out", type=str, default="lora-med")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=nf4, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)

    train_data = prepare_dataset(sample_size=4000)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        args=SFTConfig(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            logging_steps=20,
            save_steps=500,
            bf16=True,
            max_seq_length=2048,
            packing=True,
        ),
        formatting_func=lambda x: x["text"],
    )

    trainer.train()
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Done. LoRA adapters saved to: {args.out}")

if __name__ == "__main__":
    main()
