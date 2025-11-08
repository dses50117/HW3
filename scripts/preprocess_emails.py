#!/usr/bin/env python3
"""Preprocess SMS/Email CSV for spam classification.

Writes cleaned CSV to datasets/processed/sms_spam_clean.csv and optional per-step columns.
Saves label mapping to models/label_mapping.json and run metadata to reports/metrics/run_meta.json

Usage:
  python scripts\preprocess_emails.py --input-url <url> [--no-header] --label-col-index 0 --text-col-index 1 --output-text-col text --save-step-columns --steps-out-dir datasets/processed/steps --random-seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import urllib.request
from datetime import datetime
from typing import List


DEFAULT_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


SIMPLE_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "in",
    "on",
    "for",
    "of",
    "to",
    "is",
    "are",
}


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-(). ]{6,}\d)\b")
NUMBER_RE = re.compile(r"\b\d+\b")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_url(url: str, dest: str) -> None:
    with urllib.request.urlopen(url) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)


def normalize_text(text: str, do_stem: bool = False) -> str:
    # basic pipeline: lowercase, URL/email/phone/number normalization, punctuation stripping, stopword removal, optional stemming, whitespace compaction
    t = text.lower()
    t = URL_RE.sub(" <URL> ", t)
    t = EMAIL_RE.sub(" <EMAIL> ", t)
    t = PHONE_RE.sub(" <PHONE> ", t)
    t = NUMBER_RE.sub(" <NUM> ", t)
    # simple punctuation removal
    t = re.sub(r"[\"'.,:;!?()\[\]{}\\/<>@#$%^&*+=~-]", " ", t)
    tokens = [tok for tok in re.split(r"\s+", t) if tok and tok not in SIMPLE_STOPWORDS]

    if do_stem:
        try:
            from nltk.stem.porter import PorterStemmer

            stemmer = PorterStemmer()
            tokens = [stemmer.stem(tok) for tok in tokens]
        except Exception:
            # nltk not available or download needed; skip stemming
            pass

    return " ".join(tokens)


def process_rows(rows: List[List[str]], label_idx: int, text_idx: int, output_text_col: str, do_steps: bool, do_stem: bool):
    out_rows = []
    step_columns = []
    for row in rows:
        label = row[label_idx].strip()
        text = row[text_idx].strip()
        steps = {}
        steps["original"] = text
        cleaned = normalize_text(text, do_stem=do_stem)
        steps["cleaned"] = cleaned
        out_rows.append({"label": label, output_text_col: cleaned})
        if do_steps:
            step_columns.append(steps)
    return out_rows, step_columns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess SMS/email CSV for spam classification")
    parser.add_argument("--input-url", default=DEFAULT_URL, help="URL to download CSV from (default: public dataset)")
    parser.add_argument("--input-file", help="Local CSV input file (if provided, skips download)")
    parser.add_argument("--no-header", action="store_true", help="Treat CSV as having no header row")
    parser.add_argument("--label-col-index", type=int, default=0)
    parser.add_argument("--text-col-index", type=int, default=1)
    parser.add_argument("--output-text-col", default="text")
    parser.add_argument("--save-step-columns", action="store_true")
    parser.add_argument("--steps-out-dir", default=os.path.join("datasets", "processed", "steps"))
    parser.add_argument("--out-csv", default=os.path.join("datasets", "processed", "sms_spam_clean.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-stem", action="store_true", help="Attempt stemming (requires nltk)")

    args = parser.parse_args(argv)

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    ensure_dir(os.path.dirname(args.steps_out_dir) or ".")
    ensure_dir("models")
    ensure_dir(os.path.join("reports", "metrics"))

    input_path = args.input_file
    temp_downloaded = False
    if not input_path:
        input_path = os.path.join("datasets", "raw_sms_spam.csv")
        if not os.path.exists(input_path):
            print(f"Downloading dataset from {args.input_url} -> {input_path}")
            ensure_dir(os.path.dirname(input_path) or ".")
            download_url(args.input_url, input_path)
            temp_downloaded = True

    rows = []
    with open(input_path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            rows.append(r)

    if not rows:
        print("No rows found in input CSV")
        return 2

    # process
    out_rows, step_columns = process_rows(rows, args.label_col_index, args.text_col_index, args.output_text_col, args.save_step_columns, args.do_stem)

    # write cleaned csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", args.output_text_col])
        writer.writeheader()
        for r in out_rows:
            # map labels ham->0 spam->1 if possible
            lab = r["label"].lower()
            writer.writerow({"label": lab, args.output_text_col: r[args.output_text_col]})

    # write step columns if requested
    if args.save_step_columns:
        ensure_dir(args.steps_out_dir)
        steps_csv = os.path.join(args.steps_out_dir, "per_step_columns.csv")
        with open(steps_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["original", "cleaned"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in step_columns:
                writer.writerow({k: s.get(k, "") for k in fieldnames})

    # save label mapping
    label_map = {"ham": 0, "spam": 1}
    with open(os.path.join("models", "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    # save run metadata
    run_meta = {"seed": args.seed, "created_at": datetime.utcnow().isoformat() + "Z"}
    with open(os.path.join("reports", "metrics", "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print(f"Wrote cleaned CSV: {args.out_csv}")
    if args.save_step_columns:
        print(f"Wrote step columns to: {steps_csv}")
    print("Wrote models/label_mapping.json and reports/metrics/run_meta.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
