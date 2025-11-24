#!/usr/bin/env python3
"""
Quick & accurate keyphrase generator training script (T5-small or T5-base).
Now integrates standard datasets automatically:
- Default training: midas/kp20k (Hugging Face)
- Default validation: midas/inspec (Hugging Face)
You can still pass local train/validation files via --train_file and --validation_file.

Usage:
    # Quick train on a sampled subset of KP20k
    python train_quick_t5_with_datasets.py --output_dir ./t5_quick --model_name t5-small --max_train_samples 50000

    # Or use local CSV files (columns: text, keywords or title+abstract, keywords)
    python train_quick_t5_with_datasets.py --train_file ./my_train.csv --validation_file ./my_val.csv --model_name t5-base

Notes:
 - For quick iteration use t5-small; for more accuracy use t5-base.
 - Recommended to run on GPU. The script enables fp16 automatically if a CUDA GPU is available.
"""

import argparse
import os
import random
from typing import Dict, Any, Tuple, List
import numpy as np
import torch

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

# ---------- utilities ----------
def join_keywords_field(k):
    """
    Normalize keywords field which may be list[str] or str with separators.
    Return semi-colon separated string without duplicates, preserving order.
    """
    if k is None:
        return ""
    if isinstance(k, (list, tuple)):
        parts = [str(x).strip() for x in k if x and str(x).strip()]
    else:
        # string: split on common separators
        parts = [p.strip() for p in str(k).replace("\n", ";").replace("|", ";").split(";") if p.strip()]
    seen = set()
    out = []
    for p in parts:
        pl = p.lower()
        if pl not in seen:
            seen.add(pl)
            out.append(p)
    return "; ".join(out)

def detect_text_and_keyword_columns(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Heuristic to detect which columns contain text and keywords.
    Returns (text_column, keywords_column).
    """
    keys = {k.lower(): k for k in example.keys()}
    # detect keywords column names
    kw_candidates = ["keywords", "keyphrases", "keyphrase", "tags", "author_keywords", "keyword"]
    text_candidates = ["document", "text", "abstract", "title", "content", "paper", "article"]
    kw_col = None
    for c in kw_candidates:
        if c in keys:
            kw_col = keys[c]
            break
    # text column detection: prefer 'document' or 'abstract' + 'title'
    if "document" in keys:
        text_col = keys["document"]
    elif "abstract" in keys and "title" in keys:
        # we will join title + abstract during processing
        text_col = (keys["title"], keys["abstract"])
    elif "abstract" in keys:
        text_col = keys["abstract"]
    elif "title" in keys:
        text_col = keys["title"]
    else:
        # fallback to first long column
        # choose the key whose value length is largest for this example
        best = None
        best_len = -1
        for k, orig_k in keys.items():
            val = example[orig_k]
            try:
                l = len(str(val))
            except Exception:
                l = 0
            if l > best_len:
                best_len = l
                best = orig_k
        text_col = best
    if kw_col is None:
        # try to guess by checking for short list-like fields
        for k, orig_k in keys.items():
            v = example[orig_k]
            if isinstance(v, (list, tuple)) and 0 < len(v) <= 50:
                kw_col = orig_k
                break
    # final fallback
    if kw_col is None:
        kw_col = list(example.keys())[-1]
    return text_col, kw_col

def build_text_from_example(example: Dict[str, Any], text_col):
    if isinstance(text_col, tuple):
        t = ""
        for c in text_col:
            v = example.get(c, "")
            if v is None:
                v = ""
            if isinstance(v, list):
                v = " ".join(map(str, v))
            t += str(v).strip() + " "
        return t.strip()
    else:
        v = example.get(text_col, "")
        if v is None:
            return ""
        if isinstance(v, list):
            return " ".join(map(str, v))
        return str(v)

# normalize keywords string for training target
def normalize_keyword_str(s: str) -> str:
    return join_keywords_field(s)

def preprocess_examples(examples, tokenizer, text_column="text", keywords_column="keywords",
                        max_input_length=512, max_target_length=64, prefix="generate keywords: "):
    inputs = [prefix + t for t in examples[text_column]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    targets = [normalize_keyword_str(k) for k in examples[keywords_column]]
    labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    label_ids = labels["input_ids"]
    # replace pad token id's for labels with -100 so loss ignores them
    label_ids = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in label_ids]
    model_inputs["labels"] = label_ids
    return model_inputs

def compute_keyphrase_metrics(preds, labels, tokenizer):
    # preds: decoded token ids from generate; labels: token ids with -100 for pads
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def split_norm(s):
        items = [i.strip().lower() for i in s.replace("\n", ";").replace(",", ";").split(";") if i.strip()]
        return set(items)

    precs, recs, f1s = [], [], []
    for p, t in zip(decoded_preds, decoded_labels):
        pset = split_norm(p)
        tset = split_norm(t)
        tp = len(pset & tset)
        prec = tp / len(pset) if pset else 0.0
        rec = tp / len(tset) if tset else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    return {"precision": float(np.mean(precs)), "recall": float(np.mean(recs)), "f1": float(np.mean(f1s))}

# ---------- dataset helpers ----------
def load_standard_datasets(train_split: str = "train", validation_split: str = "validation", max_train_samples: int = None):
    """
    Load KP20k as default training and Inspec as default validation from Hugging Face.
    Returns a DatasetDict with 'train' and 'validation' splits.
    """
    print("Loading KP20k (midas/kp20k) for training...")
    ds_train = load_dataset("midas/kp20k", split=train_split)
    print("Loading Inspec (midas/inspec) for validation...")
    ds_val = load_dataset("midas/inspec", split=validation_split)
    # Sample subset for quick training if requested
    if max_train_samples is not None:
        n = min(max_train_samples, len(ds_train))
        print(f"Sampling {n}/{len(ds_train)} training examples for quick training...")
        ds_train = ds_train.shuffle(seed=42).select(range(n))
    # Keep as DatasetDict
    return DatasetDict({"train": ds_train, "validation": ds_val})

def convert_dataset_to_text_and_keywords(ds: Dataset):
    """
    Convert arbitrary dataset to standard columns: 'text' and 'keywords'
    by applying heuristics on columns.
    """
    # detect columns using first example
    first = ds[0]
    text_col, kw_col = detect_text_and_keyword_columns(first)

    def mapper(example):
        text = build_text_from_example(example, text_col)
        keywords = join_keywords_field(example.get(kw_col, "")) if kw_col in example else ""
        return {"text": text, "keywords": keywords}

    return ds.map(mapper, remove_columns=[c for c in ds.column_names if c not in ("text", "keywords")])

# ---------- main training flow ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=None, help="Local CSV/JSON training file with text+keywords or title+abstract+keywords")
    parser.add_argument("--validation_file", type=str, default=None, help="Local CSV/JSON validation file")
    parser.add_argument("--hf_dataset_name", type=str, default=None, help="Optional HF dataset name (overrides defaults)")
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
    parser.add_argument("--max_train_samples", type=int, default=50000, help="Sample this many training examples for quick training (None to use full dataset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed); random.seed(args.seed)

    # Load or prepare datasets
    if args.train_file or args.validation_file:
        data_files = {}
        if args.train_file:
            data_files["train"] = args.train_file
        if args.validation_file:
            data_files["validation"] = args.validation_file
        print("Loading local dataset files:", data_files)
        ds = load_dataset("csv", data_files=data_files)
        # Convert each split to standard columns
        for s in ds:
            ds[s] = convert_dataset_to_text_and_keywords(ds[s])
    elif args.hf_dataset_name:
        print("Loading dataset from Hugging Face:", args.hf_dataset_name)
        ds = load_dataset(args.hf_dataset_name)
        # try to get requested splits
        if args.train_split in ds:
            train_split = ds[args.train_split]
        else:
            train_split = ds[list(ds.keys())[0]]
        if args.validation_split in ds:
            val_split = ds[args.validation_split]
        else:
            # try to take a small sample as validation if none
            val_split = train_split.select(range(min(1000, len(train_split))))
        ds = DatasetDict({"train": train_split, "validation": val_split})
        for s in ds:
            ds[s] = convert_dataset_to_text_and_keywords(ds[s])
    else:
        # default standard datasets
        ds = load_standard_datasets(train_split=args.train_split, validation_split=args.validation_split, max_train_samples=args.max_train_samples)
        # convert to text+keywords columns
        ds["train"] = convert_dataset_to_text_and_keywords(ds["train"])
        ds["validation"] = convert_dataset_to_text_and_keywords(ds["validation"])

    # If sampling wasn't done earlier and user requested max_train_samples, apply now
    if args.max_train_samples and len(ds["train"]) > args.max_train_samples:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))

    print("Dataset splits sizes:", {k: len(ds[k]) for k in ds})

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Preprocess dataset with tokenizer
    def preprocess_fn(examples):
        # expecting columns 'text' and 'keywords' now
        texts = examples.get("text", [""] * len(examples[list(examples.keys())[0]]))
        keys = examples.get("keywords", [""] * len(texts))
        return preprocess_examples({"text": texts, "keywords": keys}, tokenizer,
                                   text_column="text", keywords_column="keywords",
                                   max_input_length=args.max_input_length, max_target_length=args.max_target_length)

    remove_cols = [c for c in ds["train"].column_names if c not in ("text", "keywords")]
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
            labels_proc = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels_proc, skip_special_tokens=True)
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

    # Train and save best model
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved trained model to:", args.output_dir)

if __name__ == "__main__":
    main()
