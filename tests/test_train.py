#!/usr/bin/env python3
import os
import subprocess
import sys
import json
import pytest

def test_train_script():
    script = os.path.join("scripts", "train_spam_classifier.py")
    cmd = [sys.executable, script, "--ngram-range", "1,2", "--min-df", "2", "--random-seed", "42"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, f"Train script failed:\n{p.stderr}"

    vec_path = os.path.join("models", "vectorizer.joblib")
    model_path = os.path.join("models", "model.joblib")
    metrics_dir = os.path.join("reports", "metrics")
    assert os.path.exists(vec_path) and os.path.exists(model_path), "Missing model artifacts"

    # basic metrics file check
    metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith("_metrics.json")]
    assert metrics_files, "No metrics JSON found in reports/metrics/"

    with open(os.path.join(metrics_dir, metrics_files[-1]), "r", encoding="utf-8") as f:
        m = json.load(f)
    for k in ("precision", "recall", "f1"):
        assert k in m, f"Metrics missing key: {k}"
