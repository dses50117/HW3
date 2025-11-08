#!/usr/bin/env python3
import os
import subprocess
import sys


def main():
    script = os.path.join("scripts", "visualize_spam.py")
    p = subprocess.run([sys.executable, script, "--token-freq"], capture_output=True, text=True)
    if p.returncode != 0:
        print("Visualize script failed:\n", p.stderr)
        return 2

    out_dir = os.path.join("reports", "visualizations")
    # Check for some expected files
    expected = ["roc.png", "pr.png", "confusion_matrix.png", "token_freq.png"]
    for fn in expected:
        if not os.path.exists(os.path.join(out_dir, fn)):
            print("Missing visualization:", fn)
            return 3

    print("PASS: visualizations created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
