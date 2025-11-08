# OpenSpec Proposal Generator (minimal)

This small tool generates an OpenSpec change proposal markdown file under `openspec/changes/specs/`.

Usage (from repository root):

```powershell
python tools\openspec-gen\generate_proposal.py --title "My Feature" --prompt "Short summary of feature"
```

The script will print the path to the created proposal file. It is dependency-free (uses only the Python standard library).

There is a test runner at `tools/openspec-gen/test_generate.py` that runs the generator and asserts the generated file contains required front-matter and sections.

To run the test:

```powershell
python tools\openspec-gen\test_generate.py
```

If you want to integrate this into CI, call the generator and then validate the produced markdown contains the headings required by your review process.
