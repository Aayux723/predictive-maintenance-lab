"""
Data loader module for UCI AI4I 2020 Predictive Maintenance Dataset.
Ensures strict feature selection and leakage prevention.
"""

import os
from typing import Tuple, List
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")

# Exactly the six permitted predictor features
FEATURE_COLS: List[str] = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET_COL: str = "Machine failure"

# Leaked or identifier columns that MUST NOT be used as predictors
EXCLUDED_LEAKAGE_COLS: List[str] = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


def load_raw_dataset(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load raw dataset from disk, downloading if missing."""
    if not os.path.exists(csv_path):
        from scripts.download_data import download_and_persist_dataset
        download_and_persist_dataset(force=False)

    df = pd.read_csv(csv_path)
    
    # Standardize column names if needed
    col_map = {
        "Air temperature": "Air temperature [K]",
        "Process temperature": "Process temperature [K]",
        "Rotational speed": "Rotational speed [rpm]",
        "Torque": "Torque [Nm]",
        "Tool wear": "Tool wear [min]",
    }
    df = df.rename(columns=col_map)
    return df


def load_clean_data(csv_path: str = DEFAULT_CSV_PATH) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load dataset and return X (strictly 6 features), y (binary target), and full raw dataframe.
    
    Returns:
        X (pd.DataFrame): 6 predictor features
        y (pd.Series): Binary target (0 = No failure, 1 = Machine failure)
        raw_df (pd.DataFrame): Full dataset for exploratory analysis
    """
    raw_df = load_raw_dataset(csv_path)

    # Validate target column presence
    if TARGET_COL not in raw_df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset: {list(raw_df.columns)}")

    # Validate all 6 features exist
    missing = [c for c in FEATURE_COLS if c not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing required predictor columns: {missing}")

    # Extract X with exact feature ordering
    X = raw_df[FEATURE_COLS].copy()
    y = raw_df[TARGET_COL].astype(int).copy()

    # Verify no leakage columns are present in X
    for col in EXCLUDED_LEAKAGE_COLS:
        if col in X.columns:
            raise ValueError(f"CRITICAL LEAKAGE: Column '{col}' found in feature matrix X!")

    return X, y, raw_df


if __name__ == "__main__":
    X, y, df = load_clean_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Columns in X: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
