"""Data loading and preprocessing utilities for the churn prediction project.

This repository intentionally contains no local SQL Server hostname, database name,
Windows username, credentials, or machine-specific connection details.
"""

from pathlib import Path
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

CATEGORICAL_COLUMNS = [
    "Gender", "Married", "State", "Value_Deal", "Phone_Service",
    "Multiple_Lines", "Internet_Service", "Internet_Type",
    "Online_Security", "Online_Backup", "Device_Protection_Plan",
    "Premium_Support", "Streaming_TV", "Streaming_Movies",
    "Streaming_Music", "Unlimited_Data", "Contract",
    "Paperless_Billing", "Payment_Method",
]

DROP_COLUMNS = ["Customer_ID", "Churn_Category", "Churn_Reason"]


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a churn dataset from a CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Place the exported churn CSV in data/raw/ and update the path."
        )

    df = pd.read_csv(path)
    logging.info("Loaded %s rows and %s columns from %s", *df.shape, path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare churn data for model training."""
    data = df.copy()

    existing_drop_columns = [c for c in DROP_COLUMNS if c in data.columns]
    data.drop(columns=existing_drop_columns, inplace=True)

    for column in CATEGORICAL_COLUMNS:
        if column in data.columns:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column].astype(str))

    if "Customer_Status" in data.columns:
        data["Customer_Status"] = data["Customer_Status"].map(
            {"Stayed": 0, "Churned": 1}
        )

    return data


def prepare_features_and_target(df: pd.DataFrame):
    """Return X and y for supervised churn modelling."""
    data = preprocess_data(df)

    if "Customer_Status" not in data.columns:
        raise ValueError("Expected target column 'Customer_Status' was not found.")

    data = data.dropna(subset=["Customer_Status"])
    X = data.drop(columns=["Customer_Status"])
    y = data["Customer_Status"].astype(int)

    return X, y
