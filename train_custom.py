#!/usr/bin/env python3
"""
Quick & accurate keyphrase generator training script (T5-small or T5-base).
- Designed for fast iteration: samples a subset of a large dataset (e.g., KP20k) for quick training.
- Saves the best model for inference.
Usage examples:
    # Basic (use local CSV with 'text' and 'keywords' columns)
    python train_quick_t5.py --train_file /path/to/kp20k_train.csv --validation_file /path/to/inspec_val.csv --output_dir ./t5_quick --model_name t5-small --max_train_samples 50000 --per_device_train_batch_size 16
    # Or using a HuggingFace dataset (if available)
    python train_quick_t5.py --hf_dataset_name your/hf-dataset --train_split train --validation_split validation --max_train_samples 30000
Notes:
 - For quick training use t5-small; for better accuracy use t5-base.
 - Recommended: run on a GPU. Use mixed precision (fp16) if available.
"""

import argparse
import os
import random
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
import numpy as np
import evaluate
import torch

# ---------- utilities ----------
def normalize_keyword_str(s: str) -> str:
    if s is None:
        return ""
    parts = [p.strip() for p in s.replace("\n", ";").replace("|", ";").replace(",", ";").split(";") if p.strip()]
    seen = set()
    out = []
    for p in parts:
        pl = p.lower()
        if pl not in seen:
            seen.add(pl)
            out.append(p)
    return "; ".join(out)

def preprocess_examples(examples, tokenizer, text_column="text", keywords_column="keywords",
                        max_input_length=512, max_target_length=64, prefix="generate keywords: "):
    inputs = [prefix + t for t in examples[text_column]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    targets = [normalize_keyword_str(k) for k in examples[keywords_column]]
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    label_ids = labels["input_ids"]
    # replace pad token id's of the labels by -100 so it's ignored by the loss
    label_ids = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in label_ids]
    model_inputs["labels"] = label_ids
    return model_inputs

def compute_keyphrase_metrics(preds, labels, tokenizer):
    # decode and compute simple P/R/F1 on phrase sets (normalized)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def split_and_norm(s):
        items = [i.strip().lower() for i in s.replace("\n",";").replace(',', ';').split(';') if i.strip()]
        return set(items)

    precs, recs, f1s = [], [], []
    for p, t in zip(decoded_preds, decoded_labels):
        pset = split_and_norm(p)
        tset = split_and_norm(t)
        tp = len(pset & tset)
        prec = tp / len(pset) if pset else 0.0
        rec = tp / len(tset) if tset else 0.0
        f1 = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    return {"precision": float(np.mean(precs)), "recall": float(np.mean(recs)), "f1": float(np.mean(f1s))}

# ---------- main training flow ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=None, help="Local CSV/JSON training file with 'text' and 'keywords' columns")
    parser.add_argument("--validation_file", type=str, default=None, help="Local CSV/JSON validation file")
    parser.add_argument("--hf_dataset_name", type=str, default=None, help="Hugging Face dataset name (optional)")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--keywords_column", type=str, default="keywords")
    parser.add_argument("--output_dir", type=str, default="./t5_quick")
    parser.add_argument("--model_name", type=str, default="t5-small", help="t5-small for speed, t5-base for quality")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_samples", type=int, default=50000, help="Sample this many training examples for quick training (set None to use full dataset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed); random.seed(args.seed)

    # Load dataset
    if args.hf_dataset_name:
        ds = load_dataset(args.hf_dataset_name)
    else:
        data_files = {}
        if args.train_file:
            data_files["train"] = args.train_file
        if args.validation_file:
            data_files["validation"] = args.validation_file
        if not data_files:
            raise ValueError("Provide --train_file or --hf_dataset_name")
        ds = load_dataset("csv", data_files=data_files)

    # Optionally sample a subset for quick training
    if args.max_train_samples and "train" in ds:
        total = len(ds["train"])
        sample_n = min(args.max_train_samples, total)
        print(f"Sampling {sample_n}/{total} training examples for quick training...")
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(sample_n))

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Preprocess dataset
    def preprocess_fn(examples):
        texts = examples.get(args.text_column, [""]*len(examples[list(examples.keys())[0]]))
        keys = examples.get(args.keywords_column, [""]*len(texts))
        return preprocess_examples({"text": texts, "keywords": keys}, tokenizer,
                                   text_column="text", keywords_column="keywords",
                                   max_input_length=args.max_input_length, max_target_length=args.max_target_length)

    remove_cols = list(ds[list(ds.keys())[0]].column_names)
    tokenized = ds.map(preprocess_fn, batched=True, remove_columns=remove_cols)

    # Data collator and training args
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        logging_steps=200,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        remove_unused_columns=True,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True
    )

    rouge = evaluate.load("rouge")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        metrics = compute_keyphrase_metrics(preds, labels, tokenizer)
        try:
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)
            metrics["rougeL"] = rouge_scores.get("rougeL", 0.0)
        except Exception:
            pass
        metrics["eval_f1"] = metrics["f1"]
        return metrics

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if "validation" in tokenized else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved trained model to:", args.output_dir)

if __name__ == "__main__":
    main()
