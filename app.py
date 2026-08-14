import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# Load saved models and preprocessing
# --------------------------------------------------

models = {
    "Logistic Regression":
        joblib.load("model/logistic_regression.pkl"),

    "Decision Tree":
        joblib.load("model/decision_tree.pkl"),

    "KNN":
        joblib.load("model/knn.pkl"),

    "Naive Bayes":
        joblib.load("model/naive_bayes.pkl"),

    "Random Forest":
        joblib.load("model/random_forest.pkl")
}

scaler = joblib.load("model/scaler.pkl")

feature_columns = joblib.load(
    "model/feature_columns.pkl"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏦 Bank Marketing Campaign Prediction")

st.write(
    """
    This application evaluates machine learning models
    trained on the Bank Marketing dataset.

    Upload the test CSV file, select a model, and view
    its performance metrics and confusion matrix.
    """
)


# --------------------------------------------------
# File upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # Read uploaded CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.write(
        f"Dataset contains {data.shape[0]} rows "
        f"and {data.shape[1]} columns."
    )

    st.dataframe(data.head())


    # --------------------------------------------------
    # Check target column
    # --------------------------------------------------

    if "y" not in data.columns:

        st.error(
            "The uploaded CSV must contain a 'y' column."
        )

    else:

        # Separate features and target
        X_input = data.drop("y", axis=1)

        y_actual = data["y"].map({
            "no": 0,
            "yes": 1
        })


        # --------------------------------------------------
        # Encode categorical variables
        # --------------------------------------------------

        X_input = pd.get_dummies(
            X_input,
            drop_first=True,
            dtype=int
        )


        # Make sure columns match training data
        X_input = X_input.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # --------------------------------------------------
        # Scale the data
        # --------------------------------------------------

        X_input_scaled = scaler.transform(
            X_input
        )


        # --------------------------------------------------
        # Model selection
        # --------------------------------------------------

        selected_model_name = st.selectbox(
            "Select a Machine Learning Model",
            list(models.keys())
        )

        selected_model = models[
            selected_model_name
        ]


        # --------------------------------------------------
        # Make predictions
        # --------------------------------------------------

        y_pred = selected_model.predict(
            X_input_scaled
        )

        y_prob = selected_model.predict_proba(
            X_input_scaled
        )[:, 1]


        # --------------------------------------------------
        # Calculate metrics
        # --------------------------------------------------

        accuracy = accuracy_score(
            y_actual,
            y_pred
        )

        precision = precision_score(
            y_actual,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_actual,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_actual,
            y_pred,
            zero_division=0
        )

        auc = roc_auc_score(
            y_actual,
            y_prob
        )

        mcc = matthews_corrcoef(
            y_actual,
            y_pred
        )


        # --------------------------------------------------
        # Display metrics
        # --------------------------------------------------

        st.subheader(
            f"Results: {selected_model_name}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )


        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )


        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_actual,
            y_pred
        )

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)


        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------

        st.subheader(
            "Classification Report"
        )

        report = classification_report(
            y_actual,
            y_pred,
            target_names=["No", "Yes"],
            zero_division=0
        )

        st.text(report)