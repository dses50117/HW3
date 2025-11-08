import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                           roc_curve, auc, precision_recall_curve, average_precision_score,
                           accuracy_score, roc_auc_score)

st.set_page_config(page_title="Spam Email Classification", layout="wide")
st.title("Spam Email Classification")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # 1. Dataset CSV
    dataset_path = st.file_uploader("1. Dataset CSV", type="csv")
    
    if dataset_path is not None:
        df = pd.read_csv(dataset_path)
        
        # 2. Label column
        label_col = st.selectbox("2. Label Column", df.columns.tolist())
        
        # 3. Text column
        text_col = st.selectbox("3. Text Column", df.columns.tolist())
        
        # 4. Models directory
        models_dir = st.text_input("4. Models Directory", value="models")
        
        # 5. Test size
        test_size = st.slider("5. Test Size", 0.1, 0.5, 0.2)
        
        # 6. Seed
        seed = st.number_input("6. Random Seed", value=42)
        
        # 7. Decision threshold
        threshold = st.slider("7. Decision Threshold", 0.0, 1.0, 0.5)




# Main content
if dataset_path is not None:
    try:
        # Load model artifacts
        vect = joblib.load(os.path.join(models_dir, "vectorizer.joblib"))
        model = joblib.load(os.path.join(models_dir, "model.joblib"))
        
        # 1. Data Overview
        st.header("1. Data Overview")
        st.write(f"Dataset shape: {df.shape}")
        
        # Class distribution
        class_dist = df[label_col].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=class_dist.index.astype(str), y=class_dist.values, ax=ax)
        ax.set_title("Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        
        # Add count labels
        for i, v in enumerate(class_dist.values):
            ax.text(i, v, str(v), ha='center', va='bottom')
        
        st.pyplot(fig)
    
    # 2. Model Performance
    st.header("2. Model Performance")
    
    # Split data
    X = df[text_col]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Transform data
    X_test_transformed = vect.transform(X_test)
    
    # Get predictions
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("F1 Score", f"{f1:.3f}")
    with col2:
        st.metric("AUC-ROC", f"{roc_auc:.3f}")
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    # 3. Live Prediction
    st.header("3. Live Prediction")
    
    text_input = st.text_area("Enter text to classify:", height=100)
    
    if st.button("Predict"):
        if text_input.strip():
            # Transform text
            X_input = vect.transform([text_input])
            
            # Make prediction
            proba = model.predict_proba(X_input)[0, 1]
            pred = int(proba >= threshold)
            
            # Display results
            st.write("Spam Probability:")
            st.progress(float(proba))
            
            # Result with styling
            if pred == 1:
                st.error(f"Prediction: SPAM (probability: {proba:.3f})")
            else:
                st.success(f"Prediction: HAM (probability: {proba:.3f})")
            
    # Add example buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Try spam example"):
            st.session_state.text_input = "URGENT! You have won a $1000 gift card. Click here to claim!"
    with col2:
        if st.button("Try ham example"):
            st.session_state.text_input = "Hey, are we still meeting for lunch tomorrow?"
    

    
    # 3. Model Performance (Test)
    st.header("📈 Model Performance")
    
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
        
        # Performance tabs
        perf_tabs = st.tabs(["Overview", "Detailed Metrics", "Threshold Analysis"])
        
        with perf_tabs[0]:
            # Key metrics
            metrics = {
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC AUC": roc_auc_score(y_test, y_prob),
                "PR AUC": average_precision_score(y_test, y_prob)
            }
            
            # Metrics display with color coding
            col1, col2, col3, col4, col5 = st.columns(5)
            metrics_display = [
                (col1, "Precision", "🎯"),
                (col2, "Recall", "🔍"),
                (col3, "F1", "⚖️"),
                (col4, "ROC AUC", "📊"),
                (col5, "PR AUC", "📉")
            ]
            
            for col, metric, emoji in metrics_display:
                value = metrics[metric]
                color = "normal"
                if value >= 0.9:
                    color = "green"
                elif value >= 0.7:
                    color = "blue"
                elif value < 0.5:
                    color = "red"
                    
                col.metric(
                    f"{emoji} {metric}",
                    f"{value:.3f}",
                    delta=None,
                    delta_color=color
                )
            
            # Confusion Matrix with annotations
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Compute percentages for annotations
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            
            # Add percentage annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j+0.5, i+0.7, f'({cm_norm[i,j]:.1%})', 
                           ha='center', va='center')
            
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.tight_layout()
            st.pyplot(fig)
        
        with perf_tabs[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC Curve
                fig, ax = plt.subplots(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
                
                # Add threshold markers
                thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
                for t in thresholds:
                    idx = (y_prob >= t).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, idx).ravel()
                    fpr_t = fp/(fp+tn)
                    tpr_t = tp/(tp+fn)
                    ax.plot(fpr_t, tpr_t, 'o', label=f'Threshold {t}')
                
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            
            with col2:
                # Precision-Recall Curve
                fig, ax = plt.subplots(figsize=(8, 6))
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                ap = average_precision_score(y_test, y_prob)
                
                ax.plot(recall, precision, color='darkgreen', lw=2,
                       label=f'PR (AP = {ap:.3f})')
                ax.fill_between(recall, precision, alpha=0.2, color='darkgreen')
                
                # Add threshold markers
                for t in thresholds:
                    idx = (y_prob >= t).astype(int)
                    p = precision_score(y_test, idx)
                    r = recall_score(y_test, idx)
                    ax.plot(r, p, 'o', label=f'Threshold {t}')
                
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title('Precision-Recall Curve')
                ax.legend(loc="lower left")
                st.pyplot(fig)
        
        with perf_tabs[2]:
            st.subheader("Threshold Analysis")
            
            # Interactive threshold selection
            test_threshold = st.slider(
                "Test different threshold values",
                min_value=0.0,
                max_value=1.0,
                value=threshold,
                step=0.05,
                key="threshold_analysis"
            )
            
            # Compute metrics for selected threshold
            y_pred_threshold = (y_prob >= test_threshold).astype(int)
            threshold_metrics = {
                "Precision": precision_score(y_test, y_pred_threshold),
                "Recall": recall_score(y_test, y_pred_threshold),
                "F1": f1_score(y_test, y_pred_threshold)
            }
            
            # Display metrics
            cols = st.columns(3)
            for col, (metric, value) in zip(cols, threshold_metrics.items()):
                col.metric(f"Threshold {test_threshold}", f"{metric}: {value:.3f}")
            
            # Show example predictions
            st.subheader("Example Predictions at Current Threshold")
            example_idx = np.random.choice(len(y_test), size=5, replace=False)
            
            example_df = pd.DataFrame({
                "Text": X_test.iloc[example_idx],
                "True Label": y_test.iloc[example_idx],
                "Predicted": y_pred_threshold[example_idx],
                "Probability": y_prob[example_idx]
            })
            
            # Style the dataframe
            def color_survived(val):
                color = 'green' if val == True else 'red'
                return f'color: {color}'
            
            styled_df = example_df.style.apply(lambda x: np.where(
                x == x.iloc[-1], 'background-color: yellow', ''), axis=1
            ).format({
                "Probability": "{:.3f}"
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading model or computing metrics: {e}")
        st.error("Make sure you have trained the model first and have the model.joblib file in your models directory")
        
    except Exception as e:
        st.error(f"Error loading model or computing metrics: {e}")
    
    # 4. Live Inference
    st.header("🔄 Live Inference")
    
    try:
        # Create two columns for the layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Example buttons with emojis
            example_col1, example_col2 = st.columns(2)
            
            with example_col1:
                if st.button("📨 Use spam example"):
                    st.session_state["text_input"] = "Free entry in 2 a wkly comp to win FA Cup final tkts"
                    
            with example_col2:
                if st.button("✉️ Use ham example"):
                    st.session_state["text_input"] = "I'll be there at 7, see you then!"

            # Text input with placeholder
            text_input = st.text_area(
                "Enter message to classify:",
                value=st.session_state.get("text_input", ""),
                key="text_input",
                height=100,
                placeholder="Type or paste a message here..."
            )
            
            # Predict button with loading state
            predict_button = st.button("🔍 Analyze Message")
            
            if predict_button and text_input.strip():
                with st.spinner("Analyzing message..."):
                    # Transform and predict
                    X_t = vect.transform([text_input])
                    prob = float(model.predict_proba(X_t)[:, 1][0])
                    pred = int(prob >= threshold)
                    
                    # Create a custom progress bar
                    st.write("Spam Probability:")
                    
                    # Custom progress bar colors
                    if prob < 0.3:
                        color = "green"
                        emoji = "✅"
                    elif prob < 0.7:
                        color = "orange"
                        emoji = "⚠️"
                    else:
                        color = "red"
                        emoji = "🚫"
                    
                    # Progress bar and prediction
                    st.progress(prob)
                    
                    # Prediction result with styling
                    st.markdown(
                        f"""
                        <div style='padding: 1rem; border-radius: 0.5rem; background-color: {"#d4edda" if not pred else "#f8d7da"}'>
                            <h3 style='margin: 0; color: {"#155724" if not pred else "#721c24"}'>
                                {emoji} Prediction: {"HAM" if not pred else "SPAM"}
                            </h3>
                            <p style='margin: 0.5rem 0 0 0;'>
                                Probability: {prob:.3f} (threshold: {threshold:.2f})
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        with col2:
            # Token importance analysis
            if text_input.strip():
                st.subheader("Token Analysis")
                
                # Get token importances
                X_t = vect.transform([text_input])
                feature_names = np.array(vect.get_feature_names_out())
                
                # Get non-zero tokens and their values
                non_zero = X_t.nonzero()[1]
                values = X_t.data
                
                # Create token importance dataframe
                tokens_df = pd.DataFrame({
                    'Token': feature_names[non_zero],
                    'TF-IDF': values
                }).sort_values('TF-IDF', ascending=False)
                
                # Display token importance
                fig, ax = plt.subplots(figsize=(8, min(10, max(4, len(tokens_df)))))
                
                colors = ['red' if v > np.mean(values) else 'blue' for v in tokens_df['TF-IDF']]
                
                sns.barplot(
                    data=tokens_df,
                    y='Token',
                    x='TF-IDF',
                    palette=colors,
                    ax=ax
                )
                
                ax.set_title('Token Importance in Message')
                plt.tight_layout()
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Error during inference: {e}")
        st.error("Make sure all model files are available in the models directory")

else:
    # Welcome message when no data is loaded
    st.markdown(
        """
        <div style='padding: 2rem; border-radius: 0.5rem; background-color: #f8f9fa; text-align: center;'>
            <h2>👋 Welcome to the Spam Email Classifier!</h2>
            <p>To get started:</p>
            <ol style='text-align: left; display: inline-block;'>
                <li>Upload your dataset CSV file using the sidebar</li>
                <li>Select the appropriate text and label columns</li>
                <li>Adjust model settings if needed</li>
                <li>Explore the analysis and make predictions!</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )




