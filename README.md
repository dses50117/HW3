# HW3 — Spam Email Classification

This repository contains a 4-phase spam email / SMS classification project:

- Data preprocessing: `scripts/preprocess_emails.py`
- Training: `scripts/train_spam_classifier.py` (TF-IDF + LogisticRegression)
- Prediction: `scripts/predict_spam.py` (single and batch modes)
- Visualization & dashboard: `scripts/visualize_spam.py` and `app/streamlit_app.py`

This README explains how to build, run tests, run the Streamlit app locally, and deploy to GitHub / Streamlit Cloud.

## Requirements

- Python 3.10
- Conda (optional but recommended)

Install dependencies (using the repo `requirements.txt`):

```powershell
C:/Users/Bill/anaconda3/Scripts/conda.exe create -p .conda python=3.10 -y
C:/Users/Bill/anaconda3/Scripts/conda.exe run -p .conda python -m pip install -r requirements.txt
```

## Quick local workflow

1. Preprocess the dataset (creates `datasets/processed/sms_spam_clean.csv`):

```powershell
C:/Users/Bill/anaconda3/Scripts/conda.exe run -p .conda python scripts/preprocess_emails.py --input-file datasets/sms_spam_no_header.csv --out-csv datasets/processed/sms_spam_clean.csv --no-header --label-col-index 0 --text-col-index 1 --output-text-col text_clean --save-step-columns --steps-out-dir datasets/processed/steps
```

2. Train the model (saves artifacts under `models/` and `reports/metrics`):

```powershell
C:/Users/Bill/anaconda3/Scripts/conda.exe run -p .conda python scripts/train_spam_classifier.py --input-csv datasets/processed/sms_spam_clean.csv --class-weight balanced --ngram-range 1,2 --min-df 2 --sublinear-tf --C 2.0 --eval-threshold 0.50
```

3. Run predictions (single and batch examples):

```powershell
C:/Users/Bill/anaconda3/Scripts/conda.exe run -p .conda python scripts/predict_spam.py --mode single --text "Free entry in 2 a wkly comp to win cash"
C:/Users/Bill/anaconda3/Scripts/conda.exe run -p .conda python scripts/predict_spam.py --mode batch --input-file datasets/processed/sms_spam_clean.csv --text-col text_clean --output-file predictions.csv
```

4. Generate visualizations:

```powershell
C:/Users/Bill/anaconda3/Scripts/conda.exe run -p .conda python scripts/visualize_spam.py --token-freq
```

5. Run Streamlit dashboard locally:

```powershell
C:/Users/Bill/anaconda3/Scripts/conda.exe run -p .conda streamlit run app/streamlit_app.py
```

## CI / GitHub

A GitHub Actions workflow is included at `.github/workflows/ci.yml` to run tests on pushes/PRs to `main`.

To publish this project to GitHub (run these from the project root):

```powershell
git init
git add .
git commit -m "Initial spam-classifier project"
git remote add origin https://github.com/dses50117/HW3.git
git branch -M main
git push -u origin main
```

Note: You'll need write access to the remote repository. If you prefer SSH, replace the remote URL accordingly.

## Deploy Streamlit App (Streamlit Cloud)

1. Push your repo to GitHub (see commands above).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app**, select the `dses50117/HW3` repo, the branch (`main`) and the file path `app/streamlit_app.py`.
4. Set environment variables if needed and deploy. Streamlit Cloud will install dependencies from `requirements.txt`.

## Notes & troubleshooting

- If native libs fail to import (e.g., NumPy C-extension errors), recreate the Conda env with a compatible Python version (3.10 recommended).
- For large datasets, reduce sample size before transforming with TF-IDF to avoid high memory usage.

If you want, I can open a terminal and run the final `git push` for you — but I won't do it without permission because it requires your credentials.
# Spam classification pipeline (OpenSpec sample)

This repository demonstrates an OpenSpec-driven pipeline for preprocessing, training, visualizing, and deploying a spam (SMS/email) classifier.

Quickstart

1. Preprocess the public SMS spam CSV:

```powershell
python scripts\preprocess_emails.py --save-step-columns
```

2. Train a model:

```powershell
python scripts\train_spam_classifier.py --ngram-range 1,2 --min-df 2 --random-seed 42
```

3. Visualize results:

```powershell
python scripts\visualize_spam.py --token-freq
```

4. Run Streamlit app:

```powershell
streamlit run app\streamlit_app.py
```

Run tests:

```powershell
python -m pytest -q
```
