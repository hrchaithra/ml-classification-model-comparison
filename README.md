# Classification Model Comparison

## a. Problem statement
Implement and compare five classification algorithms on one dataset and provide an interactive Streamlit application that evaluates labelled test data using saved models.

## b. Dataset description
**Dataset:** Dry Bean (UCI Machine Learning Repository, dataset ID 602)  
**Task:** Multiclass classification of dry bean varieties.  
**Features:** 16 numerical measurements derived from bean images.  
**Target:** Class (bean variety).  
**Source:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset  
The training script retrieves the dataset using ucimlrepo, then creates a stratified 80/20 train/test split with random state 42. The held-out test rows are written to test_data.csv.

## c. GitHub Repository Link
GitHub repository: https://github.com/hrchaithra/ml-classification-model-comparison

## d. Models used

Models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine.

## Model Comparison

The five classification models were evaluated on the held-out test dataset using Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8884 | 0.9531 | 0.8882 | 0.8884 | 0.8879 | 0.8652 |
| KNN | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9949 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

### Observations on Model Performance

- **Logistic Regression:** Achieved an accuracy of 92.07% and an AUC of 0.9948. It performed strongly across all reported metrics.
- **Decision Tree:** Achieved an accuracy of 88.84%. It had the lowest accuracy and MCC among the five evaluated models.
- **K-Nearest Neighbors (KNN):** Achieved an accuracy of 91.66% and an AUC of 0.9833, demonstrating strong classification performance.
- **Gaussian Naive Bayes:** Achieved an accuracy of 89.79% and an AUC of 0.9916. Its AUC was high despite its lower accuracy compared with the top-performing models.
- **Support Vector Machine (SVM):** Achieved the highest accuracy (92.21%), AUC (0.9949), F1-score (0.9222), and MCC (0.9059) among the evaluated models.

### Overall Best Model

**Support Vector Machine (SVM)** was the best-performing model on this test split. It achieved the highest accuracy, AUC, F1-score, and MCC of the five models. Its results indicate a strong balance of predictive performance on the Dry Bean classification task.

### Streamlit App
Deployed Streamlit app URL: https://ml-classification-model-comparison-jszjkkdzarg7dfdhx3kv9w.streamlit.app/

## Project files
- app.py — Streamlit frontend and evaluation.
- train_model.py — downloads data, trains the five classifiers, evaluates them, and saves model artifacts.
- model/ — saved.joblib bundles created by training.
- test_data.csv — labelled holdout data created by training.
- training_metrics.csv — holdout comparison results.
- requirements.txt — Python dependencies.

