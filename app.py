
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="Classification Model Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Classification Model Evaluation")
st.write(
    "Compare five trained classifiers using a labelled test dataset. "
    "Upload a CSV containing the required features and target column `Class`."
)


@st.cache_resource
def load_models():
    models = {}

    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename

        if path.exists():
            models[name] = joblib.load(path)

    return models


models = load_models()

if not models:
    st.error(
        "No saved models found. Run `python train_model.py` first."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a labelled test CSV to evaluate the models.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

data.columns = data.columns.astype(str).str.strip()

if data.columns.duplicated().any():
    st.error("The CSV contains duplicate column names.")
    st.stop()

# Validate saved model bundles
try:
    first_model = next(iter(models.values()))
    target_name = first_model.get("target_name", "Class")
    feature_names = first_model["feature_names"]

    for bundle in models.values():
        if (
            bundle["feature_names"] != feature_names
            or bundle.get("target_name", "Class") != target_name
        ):
            st.error("The saved models have inconsistent feature or target columns.")
            st.stop()

except (KeyError, TypeError) as exc:
    st.error(f"Invalid saved model format: {exc}")
    st.stop()

if target_name not in data.columns:
    st.error(f"Missing target column: {target_name}")
    st.stop()

missing = [col for col in feature_names if col not in data.columns]

if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

X = data[feature_names].apply(pd.to_numeric, errors="coerce")
y_true = data[target_name].astype("string").str.strip()

if X.isna().any().any():
    st.error("Feature columns contain missing or non-numeric values.")
    st.stop()

if not X.map(lambda value: abs(value) != float("inf")).all().all():
    st.error("Feature columns contain infinite values.")
    st.stop()

if y_true.isna().any() or y_true.eq("").any():
    st.error("The target column contains missing labels.")
    st.stop()

if y_true.nunique() < 2:
    st.error("The test dataset must contain at least two classes.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

model_name = st.selectbox(
    "Select Classifier",
    list(models.keys()),
)

pipeline = models[model_name]["pipeline"]

try:
    y_pred = pipeline.predict(X)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

y_eval = y_true.astype(str).to_numpy()
y_pred = pd.Series(y_pred).astype(str).str.strip().to_numpy()

# Evaluation metrics
metrics = {
    "Accuracy": accuracy_score(y_eval, y_pred),
    "Precision": precision_score(
        y_eval, y_pred, average="weighted", zero_division=0
    ),
    "Recall": recall_score(
        y_eval, y_pred, average="weighted", zero_division=0
    ),
    "F1-score": f1_score(
        y_eval, y_pred, average="weighted", zero_division=0
    ),
    "MCC": matthews_corrcoef(y_eval, y_pred),
}

# AUC calculation
try:
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("The selected model does not support predict_proba.")

    y_prob = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)

    if len(classes) == 2:
        auc = roc_auc_score(
            (y_true == str(classes[1])).astype(int),
            y_prob[:, 1],
        )
    else:
        auc = roc_auc_score(
            y_eval,
            y_prob,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )

    metrics["AUC"] = auc

except (ValueError, AttributeError, IndexError):
    metrics["AUC"] = None

st.subheader(f"Evaluation Metrics — {model_name}")

columns = st.columns(3)

for index, (name, value) in enumerate(metrics.items()):
    if value is not None:
        columns[index % 3].metric(name, f"{value:.4f}")
    else:
        columns[index % 3].metric(name, "N/A")

# Confusion matrix
st.subheader("Confusion Matrix")

labels = sorted(set(y_eval) | set(y_pred))
cm = confusion_matrix(y_eval, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(8, 5))
image = ax.imshow(cm, interpolation="nearest")

ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)

fig.colorbar(image, ax=ax)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Classification report
st.subheader("Classification Report")

report = classification_report(
    y_eval,
    y_pred,
    labels=labels,
    zero_division=0,
    output_dict=True,
)

st.dataframe(
    pd.DataFrame(report).transpose().round(4),
    use_container_width=True,
)
