import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Assignment 2 - Classification models", layout="wide")


ARTIFACT_DIR = os.path.join("model", "artifacts")
DEFAULT_TEST_PATH = os.path.join("data", "test_data.csv")
SCALER_FILE = "standard_scaler.joblib"

MODEL_FILES = {
    "Logistic Regression": ("logistic_regression_model.joblib", "scaled"),
    "Decision Tree": ("decision_tree_model.joblib", "unscaled"),
    "KNN (k=5)": ("knn_model.joblib", "scaled"),
    "Gaussian Naive Bayes": ("gaussian_nb_model.joblib", "scaled"),
    "Random Forest (Ensemble)": ("random_forest_model.joblib", "unscaled"),
    "XGBoost (Ensemble)": ("xgboost_model.joblib", "unscaled"),
}

DROP_TEXT_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title"]  
TARGET_COL = "label"


def check_artifacts():
    missing = []
    if not os.path.isdir(ARTIFACT_DIR):
        missing.append(f"Missing folder: {ARTIFACT_DIR}")
        return missing

    scaler_path = os.path.join(ARTIFACT_DIR, SCALER_FILE)
    if not os.path.exists(scaler_path):
        missing.append(f"Missing scaler: {scaler_path}")

    for name, (fname, _) in MODEL_FILES.items():
        fpath = os.path.join(ARTIFACT_DIR, fname)
        if not os.path.exists(fpath):
            missing.append(f"Missing model file for '{name}': {fpath}")

    return missing


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return df


def prepare_X_y(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'label' column for evaluation metrics.")

    df_clean = df.copy()

    cols_to_drop = [c for c in DROP_TEXT_COLS if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop, errors="ignore")

    y = df_clean[TARGET_COL].astype(int).values

    X = df_clean.drop(columns=[TARGET_COL], errors="ignore")

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        for c in non_numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        if X.isna().any().any():
            raise ValueError(
                f"Non-numeric columns detected and coercion introduced NaNs. "
                f"Columns: {non_numeric}. Please upload the numeric test_data.csv produced by your notebook."
            )

    return X, y


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def predict_model(model, X: pd.DataFrame, scaler=None):
    # Keep X as DataFrame to preserve column names
    X_in = X.copy()

    # Apply scaling ONLY for scaled models, and keep DataFrame after scaling
    if scaler is not None:
        X_scaled = scaler.transform(X_in)
        X_in = pd.DataFrame(X_scaled, columns=X_in.columns, index=X_in.index)

    # Case 1: sklearn models (LR, DT, KNN, GNB, RF)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_in)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        return y_proba, y_pred

    # Case 2: XGBoost native Booster (your saved model)
    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X_in, feature_names=list(X_in.columns))
        y_proba = model.predict(dmat)
        y_pred = (y_proba >= 0.5).astype(int)
        return y_proba, y_pred

    # Fallback
    y_pred = model.predict(X_in)
    y_proba = y_pred.astype(float)
    return y_proba, y_pred

    
st.title("ML Assignment 2 - Classification models)")
st.write(
    """
This app allows you to upload the **test dataset CSV** and evaluate multiple trained classifiers.
It includes:
- **Dataset upload (CSV)** (test data only)
- **Model selection dropdown**
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Confusion matrix** and **classification report**
"""
)

with st.sidebar:
    
    st.header("Model Selection")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

st.subheader("Step 1: Download Test Dataset")

if os.path.exists(DEFAULT_TEST_PATH):
    with open(DEFAULT_TEST_PATH, "rb") as f:
        st.download_button(
            label="Download test_data.csv",
            data=f,
            file_name="test_data.csv",
            mime="text/csv"
        )
else:
    st.warning("Default test_data.csv not found in ./data/ folder.")

st.subheader("Step 2: Upload Test Dataset")


uploaded = st.file_uploader("Upload test dataset (CSV) - must include 'label' column", type=["csv"])

df = None

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("Dataset uploaded successfully.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")


# Main evaluation
if df is not None:
    try:
        X, y_true = prepare_X_y(df)

        # Load selected model
        model_file, model_type = MODEL_FILES[model_name]
        model_path = os.path.join(ARTIFACT_DIR, model_file)

        if not os.path.exists(model_path):
            st.error(f"Selected model file not found: {model_path}")
            st.stop()

        model = joblib.load(model_path)

        # Load scaler if needed
        scaler = None
        if model_type == "scaled":
            scaler_path = os.path.join(ARTIFACT_DIR, SCALER_FILE)
            if not os.path.exists(scaler_path):
                st.error(f"Scaler file not found (required for {model_name}): {scaler_path}")
                st.stop()
            scaler = joblib.load(scaler_path)

        # Predict
        y_proba, y_pred = predict_model(model, X, scaler=scaler)

        # Metrics
        metrics = compute_metrics(y_true, y_pred, y_proba)

        st.subheader(f"Results: {model_name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['Accuracy']:.6f}")
        c2.metric("AUC", f"{metrics['AUC']:.6f}")
        c3.metric("MCC", f"{metrics['MCC']:.6f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Precision", f"{metrics['Precision']:.6f}")
        c5.metric("Recall", f"{metrics['Recall']:.6f}")
        c6.metric("F1", f"{metrics['F1']:.6f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=True)

        # Classification report
        st.subheader("Classification Report")
        rep = classification_report(y_true, y_pred, digits=6)
        st.code(rep)

        # Show sample predictions
        with st.expander("Show sample predictions (first 25 rows)"):
            out = X.copy()
            out["y_true"] = y_true
            out["y_pred"] = y_pred
            out["y_proba"] = y_proba
            st.dataframe(out.head(25), use_container_width=True)

    except Exception as e:
        st.error(f"Evaluation failed: {e}")
else:
    st.info("Upload a CSV file (test data) or select 'Use default test_data.csv' to begin.")