"""Streamlit evaluation dashboard for saved Dry Bean classifiers."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE 
st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

@st.cache_resource
def load_artifacts():
    loaded = {}
    for label, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[label] = joblib.load(path)
    meta_path = MODEL_DIR / "metadata.joblib"
    meta = joblib.load(meta_path) if meta_path.exists() else None
    return loaded, meta

st.title("🫘 Dry Bean Classification – Model Evaluation")
st.caption("Compare five trained classifiers on a labelled test CSV. Models are loaded from disk; this app does not retrain them.")

models, metadata = load_artifacts()
if not models or metadata is None:
    st.error("Model artifacts are missing. Run `python train.py` locally and commit the generated model/ files.")
    st.stop()

expected_features = metadata["feature_names"]
class_names = metadata["class_names"]

with st.sidebar:
    st.header("Evaluation settings")
    selected = st.selectbox("Choose classifier", list(MODEL_FILES.keys()))
    st.write("**Expected input features:**", len(expected_features))
    st.write("**Classes:**", ", ".join(class_names))

uploaded = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
if uploaded is None:
    default_path = BASE / "test_data.csv"
    if default_path.exists():
        st.info("No file uploaded. Using the project's bundled test_data.csv.")
        data = pd.read_csv(default_path)
    else:
        st.warning("Upload a CSV containing all 16 input features and the target column `Class`.")
        st.stop()
else:
    data = pd.read_csv(uploaded)

if "Class" not in data.columns:
    st.error("The uploaded CSV must contain the target column `Class` so evaluation metrics can be calculated.")
    st.stop()

missing = [c for c in expected_features if c not in data.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

X_test = data[expected_features].apply(pd.to_numeric, errors="coerce")
y_true = data["Class"].astype(str)
if X_test.isna().all(axis=1).any():
    st.error("At least one row contains no usable numeric feature values.")
    st.stop()
if not set(y_true.unique()).issubset(set(class_names)):
    st.error(f"Target values must be among: {class_names}")
    st.stop()

model = models.get(selected)
if model is None:
    st.error(f"Saved model file for {selected} is missing.")
    st.stop()

pred = model.predict(X_test)
proba = model.predict_proba(X_test)
y_bin = label_binarize(y_true, classes=class_names)
metrics = {
    "Accuracy": accuracy_score(y_true, pred),
    "AUC (macro OvR)": roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"),
    "Precision (macro)": precision_score(y_true, pred, average="macro", zero_division=0),
    "Recall (macro)": recall_score(y_true, pred, average="macro", zero_division=0),
    "F1 (macro)": f1_score(y_true, pred, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y_true, pred),
}
st.subheader(f"Results – {selected}")
st.caption(f"Evaluated {len(data):,} labelled records.")
cols = st.columns(3)
for i, (name, value) in enumerate(metrics.items()):
    cols[i % 3].metric(name, f"{value:.4f}")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y_true, pred, labels=class_names, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

with st.expander("Preview uploaded test data"):
    st.dataframe(data.head(20), use_container_width=True)
st.download_button("Download predictions CSV",
                   data.assign(Predicted_Class=pred).to_csv(index=False).encode("utf-8"),
                   file_name="dry_bean_predictions.csv", mime="text/csv")
