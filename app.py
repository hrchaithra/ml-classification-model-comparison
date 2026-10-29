from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
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

st.set_page_config(page_title="Classifier Comparison Lab", page_icon="📊", layout="wide")
st.title("📊 Classification Model Evaluation")
st.write("Compare five trained classifiers on a labelled test CSV. Models are loaded from disk; uploaded data is **not used for training**.")
st.info("Upload a test CSV containing all feature columns and the target column `Class`. Run `python train_model.py` once before deploying so the model files exist.")

@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded

models = load_models()
if not models:
    st.error("No saved model files found. Run `python train_model.py` and commit the generated model/*.joblib files.")
    st.stop()

default_csv = ROOT / "test_data.csv"
uploaded = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
elif default_csv.exists():
    st.caption("Showing the bundled test_data.csv because no file has been uploaded.")
    data = pd.read_csv(default_csv)
else:
    st.warning("Upload a test CSV to begin.")
    st.stop()

st.write("**Dataset preview**")
st.dataframe(data.head(10), use_container_width=True)

first_model = next(iter(models.values()))
target_name = first_model.get("target_name", "Class")
feature_names = first_model["feature_names"]
if target_name not in data.columns:
    st.error(f"Target column `{target_name}` is missing. Include it to calculate evaluation metrics.")
    st.stop()
missing = [c for c in feature_names if c not in data.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()
if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

X = data[feature_names].apply(pd.to_numeric, errors="coerce")
y_true = data[target_name].astype(str).str.strip()
if X.isna().all(axis=None):
    st.error("All feature values are missing or non-numeric.")
    st.stop()

model_name = st.selectbox("Choose classifier", list(models.keys()))
bundle = models[model_name]
pipeline = bundle["pipeline"]
try:
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    auc = roc_auc_score(y_true, y_prob, labels=classes, multi_class="ovr", average="macro")
except ValueError as exc:
    st.error(f"Evaluation could not be completed. Check that the target labels match the training classes and the CSV contains valid values. Details: {exc}")
    st.stop()

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC (macro OVR)": auc,
    "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
    "Recall (weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}
st.subheader(f"Evaluation metrics — {model_name}")
cols = st.columns(3)
for i, (label, value) in enumerate(metrics.items()):
    cols[i % 3].metric(label, f"{value:.4f}")

st.subheader("Confusion matrix")
labels = sorted(set(y_true) | set(y_pred))
cm = confusion_matrix(y_true, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(cm, interpolation="nearest")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels)), labels=labels)
fig.colorbar(im, ax=ax)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("Classification report"):
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.caption("AUC is macro-averaged one-vs-rest for multiclass classification. Precision, recall, and F1 are weighted averages.")
