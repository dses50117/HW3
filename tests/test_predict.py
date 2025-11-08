#!/usr/bin/env python3
import os
import subprocess
import sys
import csv


def main():
    # single
    script = os.path.join("scripts", "predict_spam.py")
    p = subprocess.run([sys.executable, script, "--mode", "single", "--text", "Free money!!!"], capture_output=True, text=True)
    if p.returncode != 0:
        print("Predict single failed:\n", p.stderr)
        return 2

    # batch
    sample = os.path.join("datasets", "processed", "sms_spam_clean.csv")
    out = "predictions_test.csv"
    p = subprocess.run([sys.executable, script, "--mode", "batch", "--input-file", sample, "--output-file", out], capture_output=True, text=True)
    if p.returncode != 0:
        print("Predict batch failed:\n", p.stderr)
        return 3

    if not os.path.exists(out):
        print("Missing predictions output")
        return 4

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "pred_label" not in reader.fieldnames or "pred_proba" not in reader.fieldnames:
            print("Output missing expected columns")
            return 5

    os.remove(out)
    print("PASS: predict single and batch succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
