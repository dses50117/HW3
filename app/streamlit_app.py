import os
import glob
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)


st.set_page_config(layout="wide")
st.title("Spam classifier explorer")


def find_processed_datasets(path: str = "datasets/processed") -> list:
    if not os.path.exists(path):
        return []
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]


@st.cache_resource
def load_model_artifacts(models_dir: str = "models") -> Optional[tuple]:
    vec_path = os.path.join(models_dir, "vectorizer.joblib")
    model_path = os.path.join(models_dir, "model.joblib")
    if os.path.exists(vec_path) and os.path.exists(model_path):
        vect = joblib.load(vec_path)
        model = joblib.load(model_path)
        return vect, model
    return None


def latest_metrics(metrics_dir: str = "reports/metrics"):
    if not os.path.exists(metrics_dir):
        return None, None
    preds = sorted([p for p in os.listdir(metrics_dir) if p.endswith("_predictions.csv")])
    mets = sorted([p for p in os.listdir(metrics_dir) if p.endswith("_metrics.json")])
    pred_path = os.path.join(metrics_dir, preds[-1]) if preds else None
    met_path = os.path.join(metrics_dir, mets[-1]) if mets else None
    return pred_path, met_path


artifacts = load_model_artifacts()

st.sidebar.header("Data & artifacts")
dataset_files = find_processed_datasets()
selected_dataset = st.sidebar.selectbox("Select processed dataset", ["-- none --"] + dataset_files)

pred_path, met_path = latest_metrics()
st.sidebar.markdown("**Latest metrics predictions:**")
st.sidebar.write(pred_path or "(none)")

st.sidebar.markdown("**Model artifacts:**")
st.sidebar.write("models/vectorizer.joblib and models/model.joblib")

st.sidebar.header("Inference")
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.50)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Single prediction / Live inference")
    if artifacts is None:
        st.warning("Model artifacts not found. Run training first.")
    else:
        vect, model = artifacts
        sample_spam = "Free entry in 2 a wkly comp to win cash. Reply WIN to claim"
        sample_ham = "I'll be there at 7, see you then"

        # Quick test buttons
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("Use spam example"):
            st.session_state.setdefault("text_in", sample_spam)
        if btn_col2.button("Use ham example"):
            st.session_state.setdefault("text_in", sample_ham)

        text = st.text_area("Message text", value=st.session_state.get("text_in", ""), key="text_in", height=150)
        run_predict = st.button("Predict")

        if run_predict and text.strip():
            X_t = vect.transform([text])
            proba = float(model.predict_proba(X_t)[:, 1][0])
            label = int(proba >= threshold)
            # probability bar with threshold marker (simple)
            st.write("Predicted probability:")
            st.progress(int(proba * 100))
            st.info(f"Label: {'spam' if label==1 else 'ham'}  —  Prob: {proba:.3f} (threshold {threshold:.2f})")

        st.markdown("---")
        st.subheader("Artifacts info")
        st.write(f"Vectorizer vocabulary size: {len(vect.vocabulary_)}")
        st.write(f"Model type: {type(model).__name__}")

with col2:
    st.header("Dataset & exploratory")
    if selected_dataset and selected_dataset != "-- none --":
        df = pd.read_csv(selected_dataset)
        st.write(f"Dataset: {os.path.basename(selected_dataset)} — {df.shape[0]} rows")
        # column pickers
        text_col = st.selectbox("Text column", options=[c for c in df.columns.tolist() if df[c].dtype == object], index=0)
        label_col = st.selectbox("Label column", options=df.columns.tolist(), index=0)

        # class distribution
        st.subheader("Class distribution")
        vc = df[label_col].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # top tokens by class
        if st.button("Show top tokens by class"):
            try:
                vect = artifacts[0]
                topn = st.slider("Top tokens per class", 5, 50, 20)
                cols = df.columns.tolist()
                classes = vc.index.tolist()
                fig, axes = plt.subplots(nrows=len(classes), ncols=1, figsize=(8, 3 * len(classes)))
                if len(classes) == 1:
                    axes = [axes]
                for ax, cls in zip(axes, classes):
                    subset = df[df[label_col] == cls][text_col].fillna("")
                    X_t = vect.transform(subset)
                    freqs = np.asarray(X_t.sum(axis=0)).ravel()
                    inv_vocab = {i: t for t, i in vect.vocabulary_.items()}
                    top_idx = np.argsort(freqs)[-topn:][::-1]
                    tokens = [inv_vocab.get(i, "") for i in top_idx]
                    vals = freqs[top_idx]
                    sns.barplot(x=vals, y=tokens, ax=ax)
                    ax.set_title(f"Top tokens for class {cls}")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not compute tokens: {e}")
    else:
        st.info("Choose a processed dataset from the sidebar to enable dataset tools.")


st.markdown("---")

st.header("Evaluation & visualizations")
preds_file = pred_path
if preds_file:
    try:
        preds = pd.read_csv(preds_file)
        y_true = preds["y_true"].values
        y_proba = preds["y_proba"].values
        # interactive threshold slider (global)
        st.subheader("Threshold sweep (interactive)")
        thresh = st.slider("Eval threshold", 0.0, 1.0, float(threshold), key="eval_thresh")
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1v = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{prec:.3f}")
        c2.metric("Recall", f"{rec:.3f}")
        c3.metric("F1", f"{f1v:.3f}")

        # confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC and PR
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax2[0].plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        ax2[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax2[0].set_xlabel("FPR")
        ax2[0].set_ylabel("TPR")
        ax2[0].legend()

        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax2[1].plot(recall_vals, precision_vals, label=f"AP={ap:.3f}")
        ax2[1].set_xlabel("Recall")
        ax2[1].set_ylabel("Precision")
        ax2[1].legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Could not load predictions file: {e}")
else:
    st.info("No predictions CSV found in reports/metrics/. Run training to generate evaluation artifacts.")
