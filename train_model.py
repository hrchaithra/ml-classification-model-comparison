"""
Train five classification models on the UCI Dry Bean dataset.
Run: python train_model.py
This downloads the dataset, creates a stratified train/test split, saves models,
writes test_data.csv, and prints the comparison metrics.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

def main():
    print("Downloading UCI Dry Bean dataset...")
    dataset = fetch_ucirepo(id=602)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()
    X.columns = [str(c).strip() for c in X.columns]
    X = X.apply(pd.to_numeric, errors="coerce")
    y = y.astype(str).str.strip()

    # Keep a fixed, stratified holdout test set for the Streamlit evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    test = X_test.copy()
    test["Class"] = y_test.to_numpy()
    test.to_csv(ROOT / "test_data.csv", index=False)

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=2))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=RANDOM_STATE))
        ]),
    }

    class_names = sorted(y.unique())
    metric_rows = []
    for display_name, pipeline in models.items():
        print(f"Training {display_name}...")
        pipeline.fit(X_train, y_train)
        joblib.dump({
            "pipeline": pipeline,
            "feature_names": list(X.columns),
            "target_name": "Class",
            "class_names": class_names,
            "model_name": display_name,
        }, MODEL_DIR / (display_name.lower().replace(" ", "_") + ".joblib"))

        prediction = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        try:
            auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro", labels=pipeline.classes_)
        except (ValueError, TypeError):
            y_bin = label_binarize(y_test, classes=pipeline.classes_)
            auc = roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr")
        metric_rows.append({
            "Model": display_name,
            "Accuracy": accuracy_score(y_test, prediction),
            "AUC": auc,
            "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0),
            "F1": f1_score(y_test, prediction, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(y_test, prediction),
        })

    results = pd.DataFrame(metric_rows).sort_values("F1", ascending=False)
    results.to_csv(ROOT / "training_metrics.csv", index=False)
    print("\nHoldout evaluation (weighted precision/recall/F1; macro one-vs-rest AUC):")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCreated model/*.joblib, test_data.csv, and training_metrics.csv")
    print("Use these measured results in your README; do not invent metric values.")

if __name__ == "__main__":
    main()
