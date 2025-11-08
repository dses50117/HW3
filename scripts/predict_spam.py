#!/usr/bin/env python3
"""Predict spam for single text or batch CSV using saved artifacts.

Outputs batch predictions CSV with `pred_label` and `pred_proba`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import joblib
import pandas as pd


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "batch"), required=True)
    parser.add_argument("--text", help="Text for single prediction")
    parser.add_argument("--input-file", help="CSV input for batch mode")
    parser.add_argument("--output-file", default="predictions.csv")
    parser.add_argument("--text-col", default="text")
    args = parser.parse_args(argv)

    vect = joblib.load(os.path.join("models", "vectorizer.joblib"))
    model = joblib.load(os.path.join("models", "model.joblib"))

    if args.mode == "single":
        if not args.text:
            print("Provide --text for single mode")
            return 2
        X = [args.text]
        X_t = vect.transform(X)
        proba = model.predict_proba(X_t)[:, 1][0]
        label = int(proba >= 0.5)
        print(json.dumps({"pred_label": label, "pred_proba": float(proba)}))
        return 0

    # batch mode
    if not args.input_file:
        print("Provide --input-file for batch mode")
        return 3
    df = pd.read_csv(args.input_file)
    if args.text_col not in df.columns:
        print(f"Text column '{args.text_col}' not found in input")
        return 4
    X = df[args.text_col].fillna("")
    X_t = vect.transform(X)
    proba = model.predict_proba(X_t)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["pred_label"] = pred
    out["pred_proba"] = proba
    out.to_csv(args.output_file, index=False)
    print(f"Wrote predictions to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
