#!/usr/bin/env python3
import os
import subprocess
import sys
import pytest

def test_visualize_script():
    script = os.path.join("scripts", "visualize_spam.py")
    p = subprocess.run([sys.executable, script, "--token-freq"], capture_output=True, text=True)
    assert p.returncode == 0, f"Visualize script failed:\n{p.stderr}"

    out_dir = os.path.join("reports", "visualizations")
    # Check for some expected files
    expected = ["roc.png", "pr.png", "confusion_matrix.png", "token_freq.png"]
    for fn in expected:
        assert os.path.exists(os.path.join(out_dir, fn)), f"Missing visualization: {fn}"
