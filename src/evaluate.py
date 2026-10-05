"""Model evaluation helpers."""

import logging
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def print_classification_metrics(y_true, y_pred) -> None:
    """Print a standard classification report."""
    logging.info("Generating classification report...")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))


def get_classification_metrics(y_true, y_pred) -> dict:
    """Return key binary-classification metrics as a dictionary."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def get_confusion_matrix(y_true, y_pred):
    """Return the confusion matrix."""
    return confusion_matrix(y_true, y_pred)
