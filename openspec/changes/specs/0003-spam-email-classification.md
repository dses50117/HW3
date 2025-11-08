---
id: 0003
title: "Spam email (SMS) classification pipeline — 4-phase plan"
status: draft
authors:
  - You
  - OpenSpec Agent
created_at: "2025-11-08T00:00:00Z"
---

# Summary

This proposal defines a reproducible, testable, and reviewable 4-phase project to build a spam (SMS/email-like text) classification pipeline. The objective is to deliver a production-ready model and tools for developers and analysts: preprocessing, model training & evaluation, visualization & reporting, and a small app + CLI for inference and reuse.

The plan uses a public dataset (Packt Publishing SMS spam CSV) as the initial data source and prescribes deterministic experiments, artifact paths, and acceptance criteria.

## Motivation

Spam classification is a common text-classification problem with clear value for user protection and automated triage. This project creates a repeatable pipeline that lets maintainers iterate safely and validate model quality with deterministic seeds, well-defined metrics, and reproducible artifacts.

## Goals

- Produce a clean, versioned dataset suitable for text classification.
- Implement a baseline classifier and iterate to reach the acceptance thresholds.
- Ship visualization and reporting so reviewers can inspect performance, error modes, and pick an operating threshold.
- Provide a small interactive app and CLI for inference and re-use.

## Non-Goals

- This proposal does not include integration with email providers, message delivery systems, or automated blocking rules. Those are out-of-scope and require additional safety reviews.
- Automated model deployment pipelines (can be added as a follow-up).

## Data Source

- Initial dataset (CSV):
  https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

  Note: the dataset is public. If you use a private dataset later, ensure compliance with privacy policies and do not send secrets to external services.

## Project contract

- Inputs: raw CSV (above), developer prompts/config; no secrets.
- Outputs: cleaned datasets, model artifacts, reports, visualization PNGs/SVGs, and a small app.
- Error modes: missing or malformed CSV must produce a clear error. Tests must be deterministic using the stored seed.

## Phases (detailed)

### Phase 1 — Baseline preprocessing pipeline

Goal: Create a deterministic, documented preprocessing pipeline and produce cleaned dataset artifacts.

Implementation:

- Script: `scripts/preprocess_emails.py`
  - CLI flags:
    - `--no-header` (bool) — if the CSV has no header row
    - `--label-col-index` (int) — 0-based index for the label column
    - `--text-col-index` (int) — 0-based index for the text column
    - `--output-text-col` (string) — name for the output text column (default: `text`)
    - `--save-step-columns` (bool) — whether to persist per-step columns
    - `--steps-out-dir` (path) — directory to write per-step columns if `--save-step-columns`

- Cleaning operations pipeline (applied in order, configurable via flags/config):
  1. Lowercasing
  2. URL normalization (replace URLs with `<URL>`)
  3. Email normalization (replace emails with `<EMAIL>`)
  4. Phone normalization (replace phone numbers with `<PHONE>`)
  5. Number masking (replace numbers with `<NUM>`)
  6. Punctuation stripping (retain whitespace)
  7. Stopword removal (configurable stopword list)
  8. Optional stemming or lemmatization (flag to enable)
  9. Whitespace compaction

- Label mapping: `ham` → 0; `spam` → 1. Persist mapping to `models/label_mapping.json`.

- Deterministic train/test split: use a fixed random seed and persist run metadata to `reports/metrics/run_meta.json` for repeatability.

Outputs:

- `datasets/processed/sms_spam_clean.csv` — main cleaned CSV with columns: `label`, `text`, plus any other required fields.
- `datasets/processed/steps/` — optional per-step columns (if `--save-step-columns`) so reviewers can inspect each cleaning step.

Acceptance criteria (Phase 1):

- `datasets/processed/sms_spam_clean.csv` exists and has the same row count as the input CSV.
- `models/label_mapping.json` exists and maps `ham`→0, `spam`→1.
- `reports/metrics/run_meta.json` contains `seed` and `created_at`.

Tests (Phase 1):

- `tests/test_preprocess.py` should run the script on the public CSV and assert output file existence, label mapping correctness, and that per-step columns are produced when requested.

Notes: Although the dataset refers to SMS messages, the pipeline is appropriate for short text emails as well. Keep anonymization placeholders (`<URL>`, `<EMAIL>`, `<PHONE>`) consistent across tools.

### Phase 2 — Model training & evaluation

Goal: Train a robust, tunable classifier (SVM baseline with TF‑IDF) and report standard metrics; provide artifacts for inference.

Design choices:

- Primary pipeline: `scripts/train_spam_classifier.py` implementing scikit-learn components:
  - `TfidfVectorizer` with flags: `--ngram-range`, `--min-df`, `--sublinear-tf`
  - Model options: `LinearSVC` (via `sklearn.svm.LinearSVC`) or `SGDClassifier(loss="hinge")` for large-scale training.
  - (Optional) To obtain probabilistic outputs for thresholding, support `LogisticRegression` or `SGDClassifier(loss="log")` as an alternative; document tradeoffs.

- CLI flags for `train_spam_classifier.py`:
  - `--class-weight` (e.g., `balanced` or dict) — weight handling for imbalanced classes
  - `--ngram-range` (e.g., `1,2`) — n-gram tuple
  - `--min-df` (int) — min document frequency
  - `--sublinear-tf` (bool)
  - `--C` (float) — regularization for LinearSVC / LogisticRegression
  - `--eval-threshold` (float) — single probability/score threshold used for reporting main metrics
  - `--random-seed` (int) — seed persisted to run meta

Persistence (artifacts):

- `models/vectorizer.joblib`
- `models/model.joblib`
- `models/label_mapping.json` (from Phase 1)

Metrics & outputs:

- Compute and persist metrics JSON under `reports/metrics/` including: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and decision scores distribution.
- Save visualization assets under `reports/visualizations/` (PNG or SVG): ROC curve, PR curve, confusion matrix heatmap, and threshold sweep CSV.
- Produce a threshold sweep CSV enumerating thresholds and metrics at each threshold for later visualization.

Acceptance criteria (Phase 2):

- Primary target: Precision ≥ 0.90 and Recall ≥ 0.93 on a held-out test set. (If using non-probabilistic SVM, use decision function for ranking and choose threshold.)
- Vectorizer vocabulary size ≥ 1,000 tokens.
- `reports/metrics/*.json` saved and include `seed` & `timestamp`.

Tests (Phase 2):

- `tests/test_train.py` should:
  - Load `models/vectorizer.joblib` and assert `len(vectorizer.vocabulary_) >= 1000`.
  - Load `models/model.joblib` and `models/label_mapping.json`.
  - Validate reported metrics JSON contains precision and recall meeting acceptance thresholds. If thresholds are not met, test fails.

Notes on thresholds and determinism:

- Use the saved `random-seed` from `reports/metrics/run_meta.json` so runs are reproducible.
- If the SVM approach can't produce probabilities, prefer `SGDClassifier(loss="log")` or `LogisticRegression` for calibrated probabilities; otherwise use decision function + calibration (Platt scaling) as a documented step.

### Phase 3 — Visualization & reporting

Goal: Provide a one-command visualization run and a final summary document for reviewers and non-technical stakeholders.

Implementation:

- Script: `scripts/visualize_spam.py` with flags:
  - `--class-dist` — saves class distribution plot
  - `--token-freq` — saves token frequency plots per class
  - `--confusion-matrix` — saves confusion matrix
  - `--roc` — saves ROC curve
  - `--pr` — saves Precision-Recall curve
  - `--threshold-sweep` — saves threshold sweep CSV/plot

Outputs:

- `reports/visualizations/*.png` or `.svg` for each requested plot
- `reports/metrics/*.csv` for threshold sweep and token-frequency counts
- `docs/FinalReport.md` — auto-populated skeleton summarizing dataset, preprocessing, model, metrics, curves, error analysis, and recommended threshold. Includes run id and artifact links.

Acceptance (Phase 3):

- All requested plots render without manual editing and are saved with filenames that include the model run id (e.g., `run-0001_roc.png`).
- `tests/test_visualize.py` verifies the expected plot files exist after running `scripts/visualize_spam.py` using the artifacts produced in Phase 2.

### Phase 4 — App, packaging & reuse

Goal: Provide an interactive Streamlit app for quick exploration and a CLI for batch/single inference.

Implementation:

- Streamlit app: `app/streamlit_app.py` with UI elements:
  - Dataset selector (choose processed datasets)
  - Class distribution and top tokens by class
  - Confusion matrix / ROC / PR (load from `reports/visualizations/` or regenerate)
  - Threshold slider (shows live Precision/Recall/F1 using loaded metrics or by computing on the held-out set)
  - Live inference text box with probability/prediction bar and threshold marker
  - Quick test buttons (e.g., run sample inputs, show error examples)

- CLI: `scripts/predict_spam.py`
  - Modes: `--mode single --text "..."` and `--mode batch --input-file path.csv --output-file predictions.csv`
  - Output columns: original input columns, `pred_label` (0/1) and `pred_proba` (float between 0-1)

- Packaging & reproducibility:
  - `requirements.txt` pinned versions
  - `README.md` with quickstart, run commands, and model artifact expectations
  - `openspec.yaml` (small spec) describing the pipeline and required artifacts for future automation
  - Optional `Makefile` targets for common tasks (preprocess, train, visualize, test, run-app)

Acceptance (Phase 4):

- `streamlit run app/streamlit_app.py` launches and all panels work with the current artifacts (models and visualizations).
- `scripts/predict_spam.py --mode batch --input-file some.csv --output-file predictions.csv` writes `predictions.csv` containing `pred_label` and `pred_proba` columns.
- `tests/test_predict.py` covers both single and batch modes and validates output format and consistency with model artifacts.

## Metrics, logging and artifact layout

- Root paths (workspace):
  - `datasets/processed/` — processed dataset files and step columns
  - `models/` — `vectorizer.joblib`, `model.joblib`, `label_mapping.json`
  - `reports/metrics/` — JSON metrics and CSV threshold sweeps; example: `reports/metrics/run_<id>_metrics.json`
  - `reports/visualizations/` — PNG/SVG plots
  - `docs/FinalReport.md` — summary
  - `tests/` — unit/integration tests for scripts

## Testing strategy

- Unit tests for preprocessing, training, visualization, and prediction scripts.
- Integration tests that run the end-to-end flow on a small sample or the full public dataset (where feasible) and verify metrics and artifact creation.
- Test files:
  - `tests/test_preprocess.py`
  - `tests/test_train.py`
  - `tests/test_visualize.py`
  - `tests/test_predict.py`

Each test run should write artifacts to a temp directory and not overwrite existing production artifacts unless explicitly requested.

## Rollout and migration notes

- No DB migrations required for this initial project. If later integrated into production systems, add migration docs and compatibility checks.
- When updating preprocessing or vectorizer vocabulary, record mapping versions and consider backward compatibility of saved models.

## Security & Privacy

- Mask or remove PII placeholders (`<EMAIL>`, `<PHONE>`, `<URL>`) in outputs intended for reports.
- Do not upload sensitive data to third-party LLMs or external services. If using an external ML service, require an explicit configuration and security review.

## Risks & Mitigations

- Risk: Model overfits to dataset and fails on real messages. Mitigation: evaluate on out-of-domain samples and include error analysis in `FinalReport.md`.
- Risk: Non-probabilistic model complicates threshold selection. Mitigation: provide calibration (Platt scaling) or use logistic regression for probabilities.
- Risk: Privacy leak in reports. Mitigation: sanitize and redact PII when producing artifacts for public review.

## Acceptance criteria (summary)

- Phase 1: cleaned dataset exists, label mapping saved, run meta with seed persisted.
- Phase 2: vectorizer vocabulary ≥ 1,000 tokens, Precision ≥ 0.90 and Recall ≥ 0.93 on held-out set, metrics and model artifacts saved.
- Phase 3: visualization artifacts generated and `FinalReport.md` populated.
- Phase 4: interactive Streamlit app works; CLI writes predictions with `pred_label` and `pred_proba` columns; tests cover both modes.

## Review checklist

- [ ] Proposal includes dataset source and provenance
- [ ] Preprocessing script with configurable steps and per-step output
- [ ] Deterministic seed saved in `reports/metrics/run_meta.json`
- [ ] Training script persists `vectorizer.joblib` and `model.joblib`
- [ ] Metrics JSON includes precision/recall/F1/ROC-AUC/PR-AUC and confusion matrix
- [ ] Threshold sweep CSV is produced
- [ ] Visualization script can produce all requested plots
- [ ] Streamlit app and CLI produce expected prediction outputs
- [ ] Tests included for each phase and pass in CI

## Example commands

Preprocess:

```powershell
python scripts\preprocess_emails.py --no-header --label-col-index 0 --text-col-index 1 --output-text-col text --save-step-columns --steps-out-dir datasets\processed\steps
```

Train:

```powershell
python scripts\train_spam_classifier.py --ngram-range 1,2 --min-df 2 --sublinear-tf --C 1.0 --class-weight balanced --eval-threshold 0.5 --random-seed 42
```

Visualize:

```powershell
python scripts\visualize_spam.py --roc --pr --confusion-matrix --threshold-sweep
```

Run Streamlit app:

```powershell
streamlit run app\streamlit_app.py
```

Batch predict:

```powershell
python scripts\predict_spam.py --mode batch --input-file to_classify.csv --output-file predictions.csv
```

## Next steps / implementation plan

1. Implement Phase 1 preprocessing script and tests. Validate on the public dataset.
2. Implement Phase 2 training script, establish artifacts and metrics, iterate to reach acceptance thresholds.
3. Implement Phase 3 visualization and generate `docs/FinalReport.md`.
4. Implement Phase 4 app and CLI, add `requirements.txt`, `README.md`, and CI to run tests and validate proposal structure.

If you approve this proposal I will scaffold the required scripts, tests, and a minimal Streamlit app in successive changes. Optionally I can start by implementing Phase 1 and its tests and open a PR for review.
