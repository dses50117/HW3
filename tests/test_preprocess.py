#!/usr/bin/env python3
"""Test for preprocess_emails.py

Runs the preprocessing script and verifies outputs are created.
"""
import os
import subprocess
import sys
import json


def main():
    script = os.path.join("scripts", "preprocess_emails.py")
    cmd = [sys.executable, script, "--save-step-columns"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("Preprocess script failed:\n", p.stderr)
        return 2

    out_csv = os.path.join("datasets", "processed", "sms_spam_clean.csv")
    label_map = os.path.join("models", "label_mapping.json")
    run_meta = os.path.join("reports", "metrics", "run_meta.json")
    steps_csv = os.path.join("datasets", "processed", "steps", "per_step_columns.csv")

    for path in (out_csv, label_map, run_meta, steps_csv):
        if not os.path.isfile(path):
            print("Missing expected output:", path)
            return 3

    # basic content checks
    with open(out_csv, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]
    if len(lines) <= 1:
        print("Cleaned CSV looks empty")
        return 4

    with open(label_map, "r", encoding="utf-8") as f:
        lm = json.load(f)
    if lm.get("ham") != 0 or lm.get("spam") != 1:
        print("Label mapping incorrect:", lm)
        return 5

    with open(run_meta, "r", encoding="utf-8") as f:
        rm = json.load(f)
    if "seed" not in rm:
        print("run_meta missing seed")
        return 6

    print("PASS: preprocess outputs created and basic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
