# Preprocessing

This document describes the preprocessing steps applied by `scripts/preprocess_emails.py`.

Pipeline:
- Lowercasing
- URL normalization -> `<URL>`
- Email normalization -> `<EMAIL>`
- Phone normalization -> `<PHONE>`
- Number masking -> `<NUM>`
- Punctuation removal
- Stopword removal (small built-in set)
- Optional stemming (requires NLTK)

Outputs:
- `datasets/processed/sms_spam_clean.csv` — cleaned dataset with `label` and `text` columns
- `datasets/processed/steps/per_step_columns.csv` — per-row original and cleaned text if requested
