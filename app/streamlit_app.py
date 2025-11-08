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

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main > div {
        padding: 2rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1v0mbdj {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with emoji
st.title("📧 Spam Email Classification")

# Sidebar configuration
with st.sidebar:
    st.header("📝 Configuration")
    
    # Dataset section
    st.subheader("Data Settings")
    dataset_path = st.file_uploader("Upload Dataset (CSV)", type="csv", help="Upload your spam/ham dataset CSV file")
    
    if dataset_path is not None:
        df = pd.read_csv(dataset_path)
        
        # Column selection
        st.write("Select columns from your dataset:")
        text_col = st.selectbox(
            "Text Column", 
            options=df.columns.tolist(),
            help="Column containing the message text"
        )
        label_col = st.selectbox(
            "Label Column", 
            options=df.columns.tolist(),
            help="Column containing spam/ham labels"
        )
        
        # Model settings
        st.subheader("Model Settings")
        models_dir = st.text_input(
            "Models Directory",
            value="models",
            help="Directory containing model artifacts"
        )
        test_size = st.slider(
            "Test Split Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of dataset to use for testing"
        )
        seed = st.number_input(
            "Random Seed",
            value=42,
            min_value=0,
            help="Seed for reproducibility"
        )
        
        # Inference settings
        st.subheader("Inference Settings")
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for spam classification"
        )




# Main content
if dataset_path is not None:
    # 1. Data Overview
    st.header("📊 Data Overview")
    
    # Dataset statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Classes", df[label_col].nunique())
    
    # Display sample data
    with st.expander("Preview Dataset"):
        st.dataframe(df.head(), use_container_width=True)
    
    # Class distribution with percentage
    st.subheader("Class Distribution")
    class_dist = df[label_col].value_counts()
    class_dist_pct = df[label_col].value_counts(normalize=True) * 100
    
    # Create a more informative distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Count plot
    sns.barplot(x=class_dist.index.astype(str), y=class_dist.values, ax=ax1, palette="viridis")
    ax1.set_title("Absolute Class Distribution")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    
    # Add count labels on bars
    for i, v in enumerate(class_dist.values):
        ax1.text(i, v, str(v), ha='center', va='bottom')
    
    # Percentage plot
    sns.barplot(x=class_dist_pct.index.astype(str), y=class_dist_pct.values, ax=ax2, palette="viridis")
    ax2.set_title("Relative Class Distribution")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Percentage (%)")
    
    # Add percentage labels on bars
    for i, v in enumerate(class_dist_pct.values):
        ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Basic text statistics
    st.subheader("Text Statistics")
    text_lengths = df[text_col].str.len()
    word_counts = df[text_col].str.split().str.len()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg. Text Length", f"{text_lengths.mean():.1f}")
    with col2:
        st.metric("Avg. Word Count", f"{word_counts.mean():.1f}")
    with col3:
        st.metric("Unique Messages", df[text_col].nunique())
        
    # Text length distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x=text_lengths, bins=50, ax=ax)
    ax.set_title("Distribution of Message Lengths")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    # 2. Top Tokens by Class
    st.header("🔍 Token Analysis")
    
    try:
        vect = joblib.load(os.path.join(models_dir, "vectorizer.joblib"))
        
        # Controls
        col1, col2 = st.columns([2, 1])
        with col1:
            topn = st.slider("Number of top tokens to show", 5, 50, 20, 
                           help="Adjust to see more or fewer tokens")
        with col2:
            normalize = st.checkbox("Normalize frequencies", value=True,
                                 help="Show relative frequencies instead of absolute counts")
        
        classes = class_dist.index.tolist()
        
        # Token analysis tabs
        tabs = st.tabs(["Token Distribution", "Word Cloud", "Comparative Analysis"])
        
        with tabs[0]:
            # Bar charts for token frequencies
            fig, axes = plt.subplots(nrows=len(classes), ncols=1, figsize=(10, 5*len(classes)))
            if len(classes) == 1:
                axes = [axes]
                
            for ax, cls in zip(axes, classes):
                subset = df[df[label_col] == cls][text_col].fillna("")
                X_t = vect.transform(subset)
                freqs = np.asarray(X_t.sum(axis=0)).ravel()
                
                if normalize:
                    freqs = freqs / len(subset)
                
                inv_vocab = {i: t for t, i in vect.vocabulary_.items()}
                top_idx = np.argsort(freqs)[-topn:][::-1]
                tokens = [inv_vocab.get(i, "") for i in top_idx]
                vals = freqs[top_idx]
                
                bars = sns.barplot(x=vals, y=tokens, ax=ax, palette="viridis")
                
                # Add value labels to bars
                for i, v in enumerate(vals):
                    if normalize:
                        ax.text(v, i, f'{v:.3f}', va='center')
                    else:
                        ax.text(v, i, f'{int(v)}', va='center')
                
                ax.set_title(f"Top {topn} tokens for class: {cls}")
                ax.set_xlabel("Normalized Frequency" if normalize else "Frequency")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tabs[1]:
            try:
                from wordcloud import WordCloud
                
                selected_class = st.selectbox("Select class for word cloud", classes)
                
                # Generate word cloud
                subset = df[df[label_col] == selected_class][text_col].fillna("")
                X_t = vect.transform(subset)
                freqs = np.asarray(X_t.sum(axis=0)).ravel()
                word_freq = {inv_vocab.get(i, ""): freq for i, freq in enumerate(freqs)}
                
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    colormap='viridis',
                                    max_words=100).generate_from_frequencies(word_freq)
                
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f"Word Cloud for class: {selected_class}")
                st.pyplot(fig)
            except ImportError:
                st.warning("WordCloud package not installed. Install with: pip install wordcloud")
        
        with tabs[2]:
            if len(classes) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    class1 = st.selectbox("First class", classes, index=0)
                with col2:
                    class2 = st.selectbox("Second class", classes, index=1)
                
                # Get frequencies for both classes
                subset1 = df[df[label_col] == class1][text_col].fillna("")
                subset2 = df[df[label_col] == class2][text_col].fillna("")
                
                X_t1 = vect.transform(subset1)
                X_t2 = vect.transform(subset2)
                
                freqs1 = np.asarray(X_t1.sum(axis=0)).ravel()
                freqs2 = np.asarray(X_t2.sum(axis=0)).ravel()
                
                if normalize:
                    freqs1 = freqs1 / len(subset1)
                    freqs2 = freqs2 / len(subset2)
                
                # Calculate frequency ratios
                epsilon = 1e-10  # To avoid division by zero
                ratios = np.log2((freqs1 + epsilon) / (freqs2 + epsilon))
                
                # Get top discriminative tokens
                top_idx = np.argsort(np.abs(ratios))[-topn:]
                tokens = [inv_vocab.get(i, "") for i in top_idx]
                vals = ratios[top_idx]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if x < 0 else 'blue' for x in vals]
                bars = ax.barh(range(len(tokens)), vals, color=colors)
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens)
                ax.set_title(f"Most discriminative tokens between {class1} and {class2}")
                ax.set_xlabel(f"Log2 ratio of frequencies ({class1} vs {class2})")
                
                # Add a legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='blue', label=f'More frequent in {class1}'),
                                 Patch(facecolor='red', label=f'More frequent in {class2}')]
                ax.legend(handles=legend_elements)
                
                st.pyplot(fig)
            else:
                st.info("Need at least 2 classes for comparative analysis")
                
    except Exception as e:
        st.error(f"Error loading vectorizer or computing tokens: {e}")
        st.error("Make sure you have trained the model first and have the vectorizer.joblib file in your models directory")
    
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




