"""Utilities for handling class imbalance with SMOTE."""

from pathlib import Path
import pandas as pd
from imblearn.over_sampling import SMOTE


def balance_dataset(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Oversample the minority class using SMOTE."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X, y)


def save_balanced_data(df: pd.DataFrame, file_path: str | Path) -> None:
    """Save a balanced dataset as CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
