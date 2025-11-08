#!/usr/bin/env python3
"""Produce visualizations from training artifacts.

Generates ROC, PR, confusion matrix and token frequency plots into reports/visualizations/.
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_curve,
                             auc, average_precision_score)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def plot_roc(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_pr(y_true, y_proba, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(out_path)
    plt.close()


def token_freq_plot(vect, model, X_series, out_path):
    # top tokens overall and by class
    vocab = vect.vocabulary_
    inv_vocab = {i: t for t, i in vocab.items()}
    X_t = vect.transform(X_series)
    freqs = np.asarray(X_t.sum(axis=0)).ravel()
    top_idx = np.argsort(freqs)[-30:][::-1]
    tokens = [inv_vocab[i] for i in top_idx]
    vals = freqs[top_idx]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=vals, y=tokens)
    plt.xlabel("TF-IDF sum")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", default=None, help="CSV with y_true,y_proba,y_pred (if not provided tries latest in reports/metrics)")
    parser.add_argument("--out-dir", default=os.path.join("reports", "visualizations"))
    parser.add_argument("--token-freq", action="store_true")
    args = parser.parse_args(argv)

    ensure_dir(args.out_dir)

    if args.predictions_csv and os.path.exists(args.predictions_csv):
        preds = pd.read_csv(args.predictions_csv)
    else:
        # find latest predictions file
        mdir = os.path.join("reports", "metrics")
        files = [f for f in os.listdir(mdir) if f.endswith("_predictions.csv")]
        if not files:
            print("No predictions file found in reports/metrics/")
            return 2
        files.sort()
        preds = pd.read_csv(os.path.join(mdir, files[-1]))

    y_true = preds["y_true"].values
    y_proba = preds["y_proba"].values
    y_pred = preds["y_pred"].values

    roc_path = os.path.join(args.out_dir, "roc.png")
    pr_path = os.path.join(args.out_dir, "pr.png")
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")

    plot_roc(y_true, y_proba, roc_path)
    plot_pr(y_true, y_proba, pr_path)
    plot_confusion(y_true, y_pred, cm_path)

    if args.token_freq:
        # load vectorizer and a sample dataset
        vect = joblib.load(os.path.join("models", "vectorizer.joblib"))
        # try to load processed dataset
        df = pd.read_csv(os.path.join("datasets", "processed", "sms_spam_clean.csv"))
        token_path = os.path.join(args.out_dir, "token_freq.png")
        token_freq_plot(vect, None, df["text"].fillna(""), token_path)

    print("Wrote visualizations to", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
