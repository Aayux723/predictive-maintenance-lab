"""
Preprocessing module for predictive maintenance lab.
Constructs ColumnTransformer and handles train/test splitting with stratification.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES: List[str] = ["Type"]
NUMERICAL_FEATURES: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs ColumnTransformer:
    - Type -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    - Numerical features -> 'passthrough' (trees are scaling-invariant)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES
            ),
            (
                "num",
                "passthrough",
                NUMERICAL_FEATURES
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified 80:20 train-test split.
    Guarantees that test distribution mirrors training distribution.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts clean, human-readable feature names from the fitted preprocessor.
    Resolves OneHotEncoder generated names cleanly (e.g., 'Type_H', 'Type_L', 'Type_M').
    """
    raw_names = list(fitted_preprocessor.get_feature_names_out())
    # Clean up any potential 'cat__' or 'num__' prefixes if present
    clean_names = []
    for name in raw_names:
        clean = name.replace("cat__", "").replace("num__", "")
        clean_names.append(clean)
    return clean_names
