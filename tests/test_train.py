#!/usr/bin/env python3
import os
import subprocess
import sys
import json


def main():
    script = os.path.join("scripts", "train_spam_classifier.py")
    cmd = [sys.executable, script, "--ngram-range", "1,2", "--min-df", "2", "--random-seed", "42"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("Train script failed:\n", p.stderr)
        return 2

    vec_path = os.path.join("models", "vectorizer.joblib")
    model_path = os.path.join("models", "model.joblib")
    metrics_dir = os.path.join("reports", "metrics")
    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        print("Missing model artifacts")
        return 3

    # basic metrics file check
    metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith("_metrics.json")]
    if not metrics_files:
        print("No metrics JSON found in reports/metrics/")
        return 4

    with open(os.path.join(metrics_dir, metrics_files[-1]), "r", encoding="utf-8") as f:
        m = json.load(f)
    for k in ("precision", "recall", "f1"):
        if k not in m:
            print("Metrics missing key:", k)
            return 5

    print("PASS: training produced artifacts and metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
