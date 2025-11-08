import os
import glob
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)


st.set_page_config(layout="wide")
st.title("Spam Email Classification")

# Sidebar inputs
st.sidebar.header("Configuration")
dataset_path = st.sidebar.file_uploader("Dataset CSV", type="csv")
if dataset_path is not None:
    df = pd.read_csv(dataset_path)
    label_col = st.sidebar.selectbox("Label column", options=df.columns.tolist())
    text_col = st.sidebar.selectbox("Text column", options=df.columns.tolist())
    models_dir = st.sidebar.text_input("Models directory", value="models")
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
    seed = st.sidebar.number_input("Random seed", value=42)
    threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)




# Main content
if dataset_path is not None:
    # 1. Data Overview
    st.header("1. Data Overview")
    st.write(f"Dataset shape: {df.shape}")
    
    # Display class distribution
    class_dist = df[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=class_dist.index.astype(str), y=class_dist.values, ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    # 2. Top Tokens by Class
    st.header("2. Top Tokens by Class")
    try:
        vect = joblib.load(os.path.join(models_dir, "vectorizer.joblib"))
        topn = st.slider("Number of top tokens to show", 5, 50, 20)
        
        classes = class_dist.index.tolist()
        fig, axes = plt.subplots(nrows=len(classes), ncols=1, figsize=(10, 4*len(classes)))
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
        st.error(f"Error loading vectorizer or computing tokens: {e}")
    
    # 3. Model Performance (Test)
    st.header("3. Model Performance (Test)")
    try:
        model = joblib.load(os.path.join(models_dir, "model.joblib"))
        
        # Split data
        X = df[text_col].fillna("")
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        # Transform and predict
        X_test_t = vect.transform(X_test)
        y_prob = model.predict_proba(X_test_t)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        # Metrics
        metrics = {
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        }
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{metrics['Precision']:.3f}")
        col2.metric("Recall", f"{metrics['Recall']:.3f}")
        col3.metric("F1", f"{metrics['F1']:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        # ROC and PR curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        ax1.plot([0, 1], [0, 1], "k--")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend()
        
        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax2.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend()
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading model or computing metrics: {e}")
    
    # 4. Live Inference
    st.header("4. Live Inference")
    try:
        col1, col2 = st.columns(2)
        
        # Example buttons
        with col1:
            if st.button("Use spam example"):
                st.session_state["text_input"] = "Free entry in 2 a wkly comp to win FA Cup final tkts"
                
        with col2:
            if st.button("Use ham example"):
                st.session_state["text_input"] = "I'll be there at 7, see you then!"

        # Text input
        text_input = st.text_area("Enter text to classify:", 
                                 value=st.session_state.get("text_input", ""),
                                 key="text_input",
                                 height=100)
        
        if st.button("Predict") and text_input.strip():
            X_t = vect.transform([text_input])
            prob = float(model.predict_proba(X_t)[:, 1][0])
            pred = int(prob >= threshold)
            
            st.write("Prediction probability:")
            st.progress(prob)
            st.info(f"Prediction: {'SPAM' if pred == 1 else 'HAM'} (probability: {prob:.3f})")
            
    except Exception as e:
        st.error(f"Error during inference: {e}")
else:
    st.info("Please upload a dataset CSV file to begin.")
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




