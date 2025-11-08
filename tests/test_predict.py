import os
import subprocess
import sys
import csv
import pytest

def test_predict_script():
    # single
    script = os.path.join("scripts", "predict_spam.py")
    p = subprocess.run([sys.executable, script, "--mode", "single", "--text", "Free money!!!"], capture_output=True, text=True)
    assert p.returncode == 0, f"Predict single failed:\n{p.stderr}"

    # batch
    sample = os.path.join("datasets", "processed", "sms_spam_clean.csv")
    out = "predictions_test.csv"
    p = subprocess.run([sys.executable, script, "--mode", "batch", "--input-file", sample, "--output-file", out], capture_output=True, text=True)
    assert p.returncode == 0, f"Predict batch failed:\n{p.stderr}"

    assert os.path.exists(out), "Missing predictions output"

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "pred_label" in reader.fieldnames and "pred_proba" in reader.fieldnames, "Output missing expected columns"

    os.remove(out)
