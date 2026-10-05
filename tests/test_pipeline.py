"""Smoke tests for the reusable churn-analysis utilities."""

import pandas as pd

from src.balance_data import balance_dataset
from src.data_preprocessing import prepare_features_and_target
from src.evaluate import get_classification_metrics, get_confusion_matrix


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer_ID": [1, 2, 3, 4],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Contract": ["Month-to-Month", "One Year", "Month-to-Month", "Two Year"],
            "Payment_Method": ["Bank", "Card", "Bank", "Card"],
            "Customer_Status": ["Stayed", "Churned", "Stayed", "Churned"],
        }
    )


def test_preprocessing_returns_model_ready_data():
    X, y = prepare_features_and_target(sample_data())
    assert "Customer_ID" not in X.columns
    assert len(X) == len(y) == 4
    assert set(y.unique()) == {0, 1}


def test_smote_balances_classes():
    X, y = prepare_features_and_target(sample_data())
    X_balanced, y_balanced = balance_dataset(X, y, random_state=42)
    assert len(X_balanced) == len(y_balanced)
    assert y_balanced.value_counts().nunique() == 1


def test_evaluation_helpers():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    metrics = get_classification_metrics(y_true, y_pred)
    matrix = get_confusion_matrix(y_true, y_pred)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert matrix.shape == (2, 2)
    assert 0 <= metrics["f1"] <= 1
