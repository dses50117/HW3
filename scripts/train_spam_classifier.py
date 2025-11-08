#!/usr/bin/env python3
"""Train a spam classifier using TF-IDF and a linear model.

Saves:
- models/vectorizer.joblib
- models/model.joblib
- reports/metrics/run_<ts>_metrics.json
- reports/metrics/run_<ts>_predictions.csv
- reports/metrics/run_<ts>_threshold_sweep.csv

Usage:
  python scripts\train_spam_classifier.py --ngram-range 1,2 --min-df 2 --sublinear-tf --C 1.0 --class-weight balanced --eval-threshold 0.5 --random-seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split


def parse_ngram(s: str):
    parts = s.split(",")
    return tuple(int(p) for p in parts)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_label_map():
    path = os.path.join("models", "label_mapping.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"ham": 0, "spam": 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram-range", default="1,1")
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--sublinear-tf", action="store_true")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", default=None)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--input-csv", default=os.path.join("datasets", "processed", "sms_spam_clean.csv"))
    args = parser.parse_args(argv)

    ensure_dir("models")
    ensure_dir(os.path.join("reports", "metrics"))
    # load data
    df = pd.read_csv(args.input_csv)
    if "label" not in df.columns or "text" not in df.columns:
        print("Input CSV must have 'label' and 'text' columns")
        return 2

    label_map = load_label_map()
    df = df.copy()
    # map labels if textual
    df["y"] = df["label"].apply(lambda x: label_map.get(str(x).lower(), int(x) if str(x).isdigit() else 0))

    X = df["text"].fillna("")
    y = df["y"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y)

    ngram_range = parse_ngram(args.ngram_range)
    vect = TfidfVectorizer(ngram_range=ngram_range, min_df=args.min_df, sublinear_tf=args.sublinear_tf)
    X_train_t = vect.fit_transform(X_train)
    X_test_t = vect.transform(X_test)

    # use LogisticRegression to get probabilities
    model = LogisticRegression(C=args.C, class_weight=args.class_weight, max_iter=1000, random_state=args.random_seed)
    model.fit(X_train_t, y_train)

    # predictions and metrics
    y_proba = model.predict_proba(X_test_t)[:, 1]
    y_pred = (y_proba >= args.eval_threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)[2]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    metrics = {
        "timestamp": ts,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "seed": args.random_seed,
    }

    metrics_path = os.path.join("reports", "metrics", f"run_{ts}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # save predictions for visualization
    preds_df = pd.DataFrame({"y_true": y_test, "y_proba": y_proba, "y_pred": y_pred})
    preds_path = os.path.join("reports", "metrics", f"run_{ts}_predictions.csv")
    preds_df.to_csv(preds_path, index=False)

    # threshold sweep
    threshs = np.linspace(0.0, 1.0, 101)
    rows = []
    for t in threshs:
        yp = (y_proba >= t).astype(int)
        prec = precision_score(y_test, yp, zero_division=0)
        rec = recall_score(y_test, yp, zero_division=0)
        f1v = precision_recall_fscore_support(y_test, yp, average="binary", zero_division=0)[2]
        rows.append({"threshold": float(t), "precision": float(prec), "recall": float(rec), "f1": float(f1v)})
    sweep_path = os.path.join("reports", "metrics", f"run_{ts}_threshold_sweep.csv")
    pd.DataFrame(rows).to_csv(sweep_path, index=False)

    # save artifacts
    joblib.dump(vect, os.path.join("models", "vectorizer.joblib"))
    joblib.dump(model, os.path.join("models", "model.joblib"))

    print("Saved vectorizer and model to models/ and metrics to reports/metrics/")
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
