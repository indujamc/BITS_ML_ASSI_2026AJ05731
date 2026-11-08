"""
Train five classifiers on the UCI Dry Bean dataset.
Run: python train.py
The script creates model/*.joblib, test_data.csv, outputs/metrics.csv,
outputs/classification_reports.txt and outputs/confusion_matrices.png.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
OUTPUT_DIR = BASE / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    path = BASE / "dry_bean.csv"
    if not path.exists():
        print("dry_bean.csv not found. Downloading from UCI...")
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=602)
        X = ds.data.features.copy()
        y = ds.data.targets.copy()
        y.columns = ["Class"]
        pd.concat([X, y], axis=1).to_csv(path, index=False)
    data = pd.read_csv(path)
    if "Class" not in data.columns:
        raise ValueError("Target column 'Class' is missing from dry_bean.csv")
    X = data.drop(columns=["Class"]).select_dtypes(include=[np.number])
    y = data["Class"].astype(str)
    if X.shape[1] < 12 or len(data) < 500:
        raise ValueError("Dataset does not meet assignment minimums (12 features, 500 rows).")
    return X, y

def build_models():
    # Scaling is essential for LR, KNN and SVM. Keeping it in each pipeline
    # ensures the scaler is fitted only on the training data.
    scaled = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", estimator)
    ])
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=42))
        ]),
        "K-Nearest Neighbors": scaled(KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "Gaussian Naive Bayes": scaled(GaussianNB()),
        "Support Vector Machine": scaled(SVC(C=3.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    }

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # The label encoder is retained for reproducibility and class ordering.
    label_encoder = LabelEncoder().fit(y_train)
    class_names = list(label_encoder.classes_)
    models = build_models()
    rows, reports, cms = [], [], {}

    for name, pipeline in models.items():
        print(f"Training {name} ...")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=class_names)
        auc = roc_auc_score(y_test_bin, proba, multi_class="ovr", average="macro")
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": auc,
            "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1": f1_score(y_test, pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred)
        }
        rows.append(row)
        reports.append(f"\n{'='*72}\n{name}\n{'='*72}\n" +
                       classification_report(y_test, pred, zero_division=0))
        cms[name] = confusion_matrix(y_test, pred, labels=class_names)
        filename_map = {
            "Logistic Regression": "logistic_regression.joblib",
            "Decision Tree": "decision_tree.joblib",
            "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
            "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
            "Support Vector Machine": "support_vector_machine.joblib",
        }
        joblib.dump(pipeline, MODEL_DIR / filename_map[name])
        print("  Accuracy:", round(row["Accuracy"], 4), "AUC:", round(row["AUC"], 4))

    metrics = pd.DataFrame(rows).sort_values("F1", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    (OUTPUT_DIR / "classification_reports.txt").write_text("\n".join(reports), encoding="utf-8")
    # Keep the true target column: the Streamlit app evaluates, not retrains.
    test = X_test.copy()
    test["Class"] = y_test.values
    test.to_csv(BASE / "test_data.csv", index=False)
    joblib.dump({"feature_names": list(X.columns), "class_names": class_names},
                MODEL_DIR / "metadata.joblib")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=False, cmap="Blues", fmt="d",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    axes[-1].axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    print("\nTEST-SET METRICS")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCreated test_data.csv, model artifacts and outputs.")

if __name__ == "__main__":
    main()
