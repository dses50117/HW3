#!/usr/bin/env python3
"""Test for preprocess_emails.py

Runs the preprocessing script and verifies outputs are created.
"""
import os
import subprocess
import sys
import json
import pytest

def test_preprocess_script():
    script = os.path.join("scripts", "preprocess_emails.py")
    cmd = [sys.executable, script, "--save-step-columns"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, f"Preprocess script failed:\n{p.stderr}"

    out_csv = os.path.join("datasets", "processed", "sms_spam_clean.csv")
    label_map = os.path.join("models", "label_mapping.json")
    run_meta = os.path.join("reports", "metrics", "run_meta.json")
    steps_csv = os.path.join("datasets", "processed", "steps", "per_step_columns.csv")

    for path in (out_csv, label_map, run_meta, steps_csv):
        assert os.path.isfile(path), f"Missing expected output: {path}"

    # basic content checks
    with open(out_csv, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) > 1, "Cleaned CSV looks empty"

    with open(label_map, "r", encoding="utf-8") as f:
        lm = json.load(f)
    assert lm.get("ham") == 0 and lm.get("spam") == 1, f"Label mapping incorrect: {lm}"

    with open(run_meta, "r", encoding="utf-8") as f:
        rm = json.load(f)
    assert "seed" in rm, "run_meta missing seed"
