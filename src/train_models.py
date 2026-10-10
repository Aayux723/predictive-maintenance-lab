"""
Model training and hyperparameter tuning module.
Tunes Decision Tree via GridSearchCV (interpretable max_depth 2-6) and Random Forest via RandomizedSearchCV.
Fits DummyClassifier baseline. Evaluates and serializes models and results.
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.data_loader import load_clean_data
from src.preprocessing import build_preprocessor, split_data, get_feature_names
from src.evaluation import (
    evaluate_pipeline,
    compute_feature_importances,
    compute_train_vs_test
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DT_PATH = os.path.join(MODELS_DIR, "decision_tree.pkl")
RF_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
DUMMY_PATH = os.path.join(MODELS_DIR, "dummy_model.pkl")
EVAL_PATH = os.path.join(MODELS_DIR, "evaluation_results.pkl")


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Trains Decision Tree with GridSearchCV optimizing average_precision over Stratified 5-Fold CV.
    Constrains max_depth to [2, 3, 4, 5, 6] for maximum human interpretability.
    """
    print("\n--- Tuning Decision Tree Classifier (GridSearchCV) ---")
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [2, 3],
        "model__min_samples_split": [10, 20],
        "model__min_samples_leaf": [5, 10],
        "model__class_weight": ["balanced", None]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    print(f"Decision Tree Best Parameters: {grid_search.best_params_}")
    print(f"Decision Tree Best CV Average Precision: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Trains Random Forest with RandomizedSearchCV (12 iterations, 3-fold stratified CV).
    Optimizes average_precision for rare failure events.
    """
    print("\n--- Tuning Random Forest Classifier (RandomizedSearchCV) ---")
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    param_dist = {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [3, 5, 8, 12, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": ["balanced", "balanced_subsample", None]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    rand_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rand_search.fit(X_train, y_train)
    print(f"Random Forest Best Parameters: {rand_search.best_params_}")
    print(f"Random Forest Best CV Average Precision: {rand_search.best_score_:.4f}")
    return rand_search.best_estimator_, rand_search.best_params_, rand_search.best_score_


def train_dummy_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Trains DummyClassifier (majority class baseline).
    Demonstrates that naive accuracy (~96.6%) is deceptive for imbalanced data.
    """
    print("\n--- Fitting Dummy Majority-Class Baseline ---")
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def run_training_pipeline() -> None:
    """
    End-to-end training, hyperparameter optimization, evaluation, and artifact serialization.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("1. Loading verified AI4I 2020 dataset...")
    X, y, raw_df = load_clean_data()

    print("2. Performing stratified 80:20 train-test split...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)} (Failures: {int(y_train.sum())})")
    print(f"Test size:  {len(X_test)} (Failures: {int(y_test.sum())})")

    # Train Models
    dt_pipeline, dt_best_params, dt_cv_score = train_decision_tree(X_train, y_train)
    rf_pipeline, rf_best_params, rf_cv_score = train_random_forest(X_train, y_train)
    dummy_pipeline = train_dummy_baseline(X_train, y_train)

    # Feature names
    fitted_prep = dt_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(fitted_prep)
    print(f"\nEngineered features ({len(feature_names)}): {feature_names}")

    # Evaluate models on untouched test set
    print("\n3. Evaluating models on untouched test set...")
    dummy_eval = evaluate_pipeline(dummy_pipeline, X_test, y_test)
    dt_eval = evaluate_pipeline(dt_pipeline, X_test, y_test)
    rf_eval = evaluate_pipeline(rf_pipeline, X_test, y_test)

    # Importances
    dt_importances = compute_feature_importances(dt_pipeline, X_test, y_test, feature_names)
    rf_importances = compute_feature_importances(rf_pipeline, X_test, y_test, feature_names)

    # Train vs Test (Overfitting check)
    dt_train_vs_test = compute_train_vs_test(dt_pipeline, X_train, y_train, X_test, y_test)
    rf_train_vs_test = compute_train_vs_test(rf_pipeline, X_train, y_train, X_test, y_test)

    # Comparison summary table
    comparison_df = pd.DataFrame([
        {"Model": "Baseline (Dummy)", **dummy_eval["metrics"]},
        {"Model": "Decision Tree", **dt_eval["metrics"]},
        {"Model": "Random Forest", **rf_eval["metrics"]}
    ])
    print("\n--- Test Set Model Comparison ---")
    print(comparison_df.to_string(index=False))

    # Persist Models
    print(f"\n4. Saving trained model pipelines to {MODELS_DIR}...")
    joblib.dump(dt_pipeline, DT_PATH)
    joblib.dump(rf_pipeline, RF_PATH)
    joblib.dump(dummy_pipeline, DUMMY_PATH)

    # Persist Evaluation Package
    eval_package = {
        "comparison_table": comparison_df,
        "dummy": {"eval": dummy_eval},
        "decision_tree": {
            "eval": dt_eval,
            "best_params": dt_best_params,
            "cv_score": dt_cv_score,
            "importances": dt_importances,
            "train_vs_test": dt_train_vs_test
        },
        "random_forest": {
            "eval": rf_eval,
            "best_params": rf_best_params,
            "cv_score": rf_cv_score,
            "importances": rf_importances,
            "train_vs_test": rf_train_vs_test
        },
        "feature_names": feature_names,
        "test_data": {
            "X_test": X_test,
            "y_test": y_test
        },
        "train_data": {
            "X_train": X_train,
            "y_train": y_train
        }
    }
    joblib.dump(eval_package, EVAL_PATH)
    print("Training and artifact serialization completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()
