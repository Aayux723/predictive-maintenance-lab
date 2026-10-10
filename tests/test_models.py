"""
Comprehensive test suite for Predictive Maintenance Explainability Lab.
Validates:
1. Dataset integrity, shape, and leakage prevention (exactly 6 predictors).
2. Binary target validity.
3. Preprocessing ColumnTransformer behavior and feature mapping.
4. Model prediction output and probability properties.
5. Tree explainer root-to-leaf path verification against scikit-learn decision_path.
6. Sensor manual input structure compatibility.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    load_clean_data,
    FEATURE_COLS,
    TARGET_COL,
    EXCLUDED_LEAKAGE_COLS
)
from src.preprocessing import (
    build_preprocessor,
    split_data,
    get_feature_names
)
from src.tree_explainer import (
    trace_sample_path,
    extract_node_data,
    explain_gini_calculation
)


@pytest.fixture(scope="session")
def clean_dataset():
    """Session fixture loading clean dataset."""
    X, y, raw_df = load_clean_data()
    return X, y, raw_df


@pytest.fixture(scope="session")
def trained_models():
    """Session fixture loading trained pipelines."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    dt = joblib.load(os.path.join(models_dir, "decision_tree.pkl"))
    rf = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    dummy = joblib.load(os.path.join(models_dir, "dummy_model.pkl"))
    return {"dt": dt, "rf": rf, "dummy": dummy}


# --- 1. DATASET TESTS ---

def test_dataset_predictors_count_and_names(clean_dataset):
    """Verify exactly the 6 permitted features are present in X."""
    X, y, raw_df = clean_dataset
    assert list(X.columns) == FEATURE_COLS, "X must contain exactly the 6 specified predictors in order."
    assert len(X.columns) == 6, f"Expected 6 features, found {len(X.columns)}"


def test_leakage_columns_excluded(clean_dataset):
    """Verify no leakage identifiers or auxiliary failure causes exist in X."""
    X, y, raw_df = clean_dataset
    for col in EXCLUDED_LEAKAGE_COLS:
        assert col not in X.columns, f"CRITICAL LEAKAGE: Column '{col}' found in feature matrix X!"


def test_binary_target(clean_dataset):
    """Verify target strictly contains only {0, 1} with no nulls."""
    X, y, raw_df = clean_dataset
    unique_vals = set(y.unique())
    assert unique_vals == {0, 1}, f"Target must be binary {0, 1}, got {unique_vals}"
    assert y.isnull().sum() == 0, "Target contains missing values."
    assert len(X) == len(y), "Features and target row count mismatch."


def test_train_test_stratification(clean_dataset):
    """Verify stratified split preserves target failure proportion."""
    X, y, raw_df = clean_dataset
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    total_prev = float(y.mean())
    train_prev = float(y_train.mean())
    test_prev = float(y_test.mean())

    assert abs(train_prev - total_prev) < 0.005, "Train set class distribution deviates from overall."
    assert abs(test_prev - total_prev) < 0.005, "Test set class distribution deviates from overall."
    assert len(X_train) == 8000
    assert len(X_test) == 2000


# --- 2. PREPROCESSING & FEATURE NAMES ---

def test_preprocessor_transformation(clean_dataset):
    """Verify ColumnTransformer correctly encodes 'Type' and passes through numeric sensor measurements."""
    X, y, raw_df = clean_dataset
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    names = get_feature_names(preprocessor)

    # 3 categories for Type ('H', 'L', 'M') + 5 numerical features = 8 total columns
    assert X_trans.shape[1] == 8, f"Expected 8 transformed features, got {X_trans.shape[1]}"
    assert len(names) == 8
    assert "Type_H" in names or "Type_L" in names
    assert "Torque [Nm]" in names
    assert "Tool wear [min]" in names


# --- 3. MODEL INFERENCE & PROBABILITIES ---

def test_model_predictions_and_probabilities(clean_dataset, trained_models):
    """Verify all models predict valid classes and normalized probability distributions."""
    X, y, raw_df = clean_dataset
    sample = X.iloc[:10]

    for model_name, pipeline in trained_models.items():
        preds = pipeline.predict(sample)
        assert len(preds) == 10
        assert set(preds).issubset({0, 1}), f"Model {model_name} generated non-binary predictions."

        if hasattr(pipeline, "predict_proba"):
            probas = pipeline.predict_proba(sample)
            assert probas.shape == (10, 2), f"Probabilities shape mismatch for {model_name}"
            # Probabilities must be in [0, 1] and sum to 1
            assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
            np.testing.assert_allclose(probas.sum(axis=1), np.ones(10), atol=1e-5)


# --- 4. TREE EXPLAINER & DECISION PATH VERIFICATION ---

def test_tree_explainer_path_accuracy(clean_dataset, trained_models):
    """
    CRITICAL TEST:
    Verify that trace_sample_path matches the exact scikit-learn decision_path()
    and correctly follows the binary tree conditions from root to leaf.
    """
    X, y, raw_df = clean_dataset
    dt_pipeline = trained_models["dt"]
    sample_df = X.iloc[[0]]  # First test row

    path_info = trace_sample_path(dt_pipeline, sample_df)

    # Scikit-learn ground truth
    preprocessor = dt_pipeline.named_steps["preprocessor"]
    tree_model = dt_pipeline.named_steps["model"]
    X_trans = preprocessor.transform(sample_df)

    expected_path = list(tree_model.decision_path(X_trans).indices)
    expected_leaf = int(tree_model.apply(X_trans)[0])

    assert path_info["visited_nodes"] == expected_path, (
        f"Visited nodes {path_info['visited_nodes']} did not match scikit-learn decision_path {expected_path}"
    )
    assert path_info["leaf_id"] == expected_leaf, (
        f"Leaf node {path_info['leaf_id']} did not match scikit-learn apply() {expected_leaf}"
    )
    assert path_info["visited_nodes"][-1] == path_info["leaf_id"], "Last visited node must be the final leaf."

    # Validate step-by-step logic
    for step in path_info["steps"]:
        actual_val = step["actual_value"]
        thresh = step["threshold"]
        expected_eval = (actual_val <= thresh)
        assert step["evaluation"] == expected_eval, "Step evaluation mismatch."
        expected_branch = "LEFT" if expected_eval else "RIGHT"
        assert step["branch"] == expected_branch, "Branch selection mismatch."


def test_gini_impurity_calculation(trained_models):
    """Verify that Gini calculation accurately matches formula 1 - sum(p_i^2)."""
    dt_pipeline = trained_models["dt"]
    tree_model = dt_pipeline.named_steps["model"]
    nodes_info = extract_node_data(tree_model, ["Type_H", "Type_L", "Type_M", "Air", "Proc", "Speed", "Torque", "Wear"])

    root_data = nodes_info[0]
    calc = explain_gini_calculation(root_data)

    p0, p1 = calc["p0"], calc["p1"]
    manual_gini = 1.0 - (p0**2 + p1**2)

    assert abs(calc["gini"] - manual_gini) < 1e-6
    assert abs(calc["p0"] + calc["p1"] - 1.0) < 1e-6


# --- 5. MANUAL INPUT COMPATIBILITY ---

def test_manual_prediction_input_format(clean_dataset, trained_models):
    """Verify that manual synthetic sensor dictionary constructs a valid DataFrame and predicts."""
    X, y, raw_df = clean_dataset
    manual_input = pd.DataFrame([{
        "Type": "M",
        "Air temperature [K]": 300.5,
        "Process temperature [K]": 310.2,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 120
    }])

    assert list(manual_input.columns) == FEATURE_COLS

    for name, pipeline in trained_models.items():
        pred = pipeline.predict(manual_input)
        assert pred[0] in [0, 1]
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(manual_input)
            assert proba.shape == (1, 2)
