# ⚙️ Predictive Maintenance Explainability Lab

An academic Machine Learning Lab project predicting industrial equipment failures using sensor telemetry and visually explaining step-by-step how a Decision Tree reaches its classifications.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-teal.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description & Leakage Prevention](#dataset-description--leakage-prevention)
3. [System Architecture](#system-architecture)
4. [Machine Learning Methodology](#machine-learning-methodology)
5. [The Core Innovation: Inside the Decision Tree](#the-core-innovation-inside-the-decision-tree)
6. [Empirical Experimental Results](#empirical-experimental-results)
7. [Installation & Reproduction](#installation--reproduction)
8. [Project Structure](#project-structure)
9. [Known Limitations](#known-limitations)
10. [Academic References](#academic-references)

---

## 1. Project Overview

In heavy manufacturing and chemical processing industries, unscheduled machine downtime generates billions of dollars in losses annually. Predictive maintenance (PdM) leverages multi-modal sensor telemetry (temperatures, torques, rotational speeds, and tool degradation) to detect imminent equipment breakdown before catastrophic failure occurs.

However, high predictive accuracy is insufficient in safety-critical manufacturing: plant engineers and operators must understand **why** an alarm was raised. Black-box models erode operator trust. 

This lab project builds a transparent machine learning system that:
- Accurately classifies machine failure vs. normal operation on severely imbalanced real-world data (3.39% failure incidence).
- Bridges the gap between predictive performance and explainability by implementing an **interactive Plotly Decision Tree visualizer** that computes exact root-to-leaf decision trajectories for any machine telemetry observation.
- Demonstrates rigorous ML engineering: leakage prevention, stratified validation, average precision optimization, cost-asymmetric confusion matrix analysis, and permutation feature importance.

---

## 2. Dataset Description & Leakage Prevention

The project utilizes the official **UCI AI4I 2020 Predictive Maintenance Dataset** (UCI ID: 601), reflecting real-world milling machine operations with 10,000 synthetic observations.

### Permitted Input Predictors (Exactly 6 Features)
| Feature Name | Type | Physical Units | Description & Observed Range |
| :--- | :--- | :--- | :--- |
| `Type` | Categorical | Quality Variant | Machine quality grade: `L` (Low, 50%), `M` (Medium, 30%), `H` (High, 20%). |
| `Air temperature [K]` | Numerical | Kelvin ($K$) | Ambient environmental air temperature ($295.3 - 304.5\ K$). |
| `Process temperature [K]` | Numerical | Kelvin ($K$) | Internal machine process temperature ($305.7 - 313.8\ K$). |
| `Rotational speed [rpm]` | Numerical | Revolutions/min | Spindle angular velocity ($1168 - 2886\ \text{rpm}$). |
| `Torque [Nm]` | Numerical | Newton-meters | Spindle resistance torque ($3.8 - 76.6\ \text{Nm}$). |
| `Tool wear [min]` | Numerical | Minutes | Accumulated cutting tool operational usage ($0 - 253\ \text{min}$). |

### Target Variable
- **`Machine failure`**: Binary indicator ($0$ = Healthy Operation, $1$ = Machine Failure).
- **Target Distribution**: $9,661$ Normal ($96.61\%$), $339$ Failures ($3.39\%$).

### Strict Prevention of Data Leakage
The raw AI4I dataset contains metadata and failure-type indicators that must **NEVER** be fed into the model:
- `UDI` and `Product ID`: Row and serial identifiers.
- `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), `RNF` (Random Failure): These auxiliary labels are direct causal subsets of `Machine failure`. Including them causes severe target leakage, fabricating 100% artificial accuracy. Our pipeline strictly filters and validates that only the 6 raw sensor features are used.

---

## 3. System Architecture

The project is structured modularly following software engineering and data science best practices:

```
predictive-maintenance-lab/
├── app.py                     # Streamlit Industrial Analytics & Explainer Dashboard
├── requirements.txt           # Verified Python package dependencies
├── README.md                  # Comprehensive technical documentation
├── project_report.md          # Full academic lab report with empirical metrics
├── data/
│   └── ai4i2020.csv           # Persisted UCI dataset (10,000 rows)
├── models/
│   ├── decision_tree.pkl      # Tuned Decision Tree Pipeline (interpretable depth)
│   ├── random_forest.pkl      # Tuned Random Forest Pipeline
│   ├── dummy_model.pkl        # Majority-class Baseline Pipeline
│   └── evaluation_results.pkl # Pre-computed test metrics, ROC/PR curves, and importances
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # UCI fetcher + CSV fallback + leakage shield
│   ├── preprocessing.py      # ColumnTransformer, OneHotEncoder, Stratified split
│   ├── train_models.py        # GridSearchCV & RandomizedSearchCV training scripts
│   ├── evaluation.py          # Metrics, confusion matrices, ROC/PR curves, permutation imp.
│   ├── tree_explainer.py      # Scikit-learn decision_path() extraction & Plotly visualizer
│   └── visualizations.py      # Industrial dark-theme charts, styling, and metric cards
├── scripts/
│   └── download_data.py       # Standalone dataset acquisition script
└── tests/
    └── test_models.py         # Pytest verification suite (9 tests covering data, models, paths)
```

---

## 4. Machine Learning Methodology

### 4.1 Preprocessing Pipeline
- **Categorical Feature (`Type`)**: Encoded into one-hot binary columns (`Type_H`, `Type_L`, `Type_M`) using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- **Numerical Features**: Passed through without rescaling. Tree-based partition algorithms are monotonically invariant to feature scale; unnecessary scaling obscures physical units ($K, \text{rpm}, \text{Nm}, \text{min}$).
- **Train-Test Partition**: $80:20$ stratified split (`random_state=42`) yielding $8,000$ training samples ($271$ failures) and $2,000$ test samples ($68$ failures). Preprocessing is fitted **only on the training split** to prevent train-test leakage.

### 4.2 Model Exploration & Hyperparameter Optimization
Three distinct models were developed:
1. **Baseline Model (`DummyClassifier`)**:
   - Strategy: `most_frequent` (always predicts $y = 0$).
   - Demonstrates the **Accuracy Paradox**: achieves $96.60\%$ accuracy while failing to detect a single breakdown ($\text{Recall} = 0.00\%$).
2. **Decision Tree Classifier (`DecisionTreeClassifier`)**:
   - Tuned via `GridSearchCV` over 5-fold `StratifiedKFold`.
   - Max depth search range constrained strictly to $[2, 3, 4, 5, 6]$ to ensure the resulting tree remains human-interpretable in safety-critical settings.
   - Evaluated across `criterion` (`gini`, `entropy`), `min_samples_split`, `min_samples_leaf`, and `class_weight` (`balanced`, `None`).
   - Optimization Metric: **Average Precision (PR-AUC)**.
3. **Random Forest Classifier (`RandomForestClassifier`)**:
   - Ensemble of de-correlated decision trees tuned via `RandomizedSearchCV` (12 iterations, 3-fold stratified CV).
   - Tuned across `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `class_weight`.

---

## 5. The Core Innovation: Inside the Decision Tree

Rather than using approximate model-agnostic surrogate methods (e.g. LIME/SHAP) that approximate black boxes with local linear regressions, our **Tree Explainer** interacts directly with scikit-learn's internal C-structures:
- `decision_path(X)`: Sparse indicator matrix of nodes visited.
- `apply(X)`: Exact leaf node assignment.
- `tree_`: Node thresholds, split feature indices, impurities, and sample counts.

### How the Decision Traversal Works
When a test sample or live manual telemetry input is evaluated:
1. The sample is transformed through the fitted `preprocessor`. Feature indices are translated to clean labels (`Torque [Nm]`, `Type_L`, etc.) via `get_feature_names_out()`.
2. The tree is traversed from Node $0$ (root) to the terminating leaf:
   $$\text{Condition: } x_j \le \theta_j$$
   - If evaluated to **TRUE**, the sample branches **LEFT**.
   - If evaluated to **FALSE**, the sample branches **RIGHT**.
3. Every step generates an explanatory card reporting:
   - Condition tested
   - Actual sensor reading
   - Boolean outcome (`TRUE` / `FALSE`)
   - Branch taken (`LEFT` / `RIGHT`)
   - Node Gini impurity and training sample count
   - Impurity reduction gain ($\Delta G$)
4. **Interactive Plotly Graph**:
   - Traversed nodes and edges are dynamically illuminated with glowing cyan/gold accents and enlarged markers.
   - Unvisited branches are dimmed to $25\%$ opacity.
   - The final leaf is marked with a distinctive golden diamond/star.

### The Mathematics of Gini Impurity
For a binary classification node containing $N$ samples ($N_0$ normal, $N_1$ failure):

$$p_0 = \frac{N_0}{N}, \quad p_1 = \frac{N_1}{N}$$

$$\text{Gini Impurity: } G = 1 - \sum_{i=0}^{1} p_i^2 = 1 - (p_0^2 + p_1^2)$$

When evaluating a split of parent node $P$ into left child $L$ and right child $R$:

$$\Delta G = G_P - \left( \frac{N_L}{N_P} G_L + \frac{N_R}{N_P} G_R \right)$$

The Decision Tree greedily selects the feature and threshold that maximizes $\Delta G$.

---

## 6. Empirical Experimental Results

All models were evaluated on the **same untouched 2,000-sample test set** ($68$ actual failures).

### Comparative Test Set Metrics
| Model | Accuracy | Precision | Recall | F1-Score | Balanced Acc | ROC-AUC | PR-AUC (Avg Prec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Dummy)** | 0.9660 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.0340 |
| **Decision Tree (Simple, 15 nodes)** | 0.9725 | 0.7826 | 0.2647 | 0.3956 | 0.6311 | 0.9056 | 0.4381 |
| **Random Forest** | **0.9860** | **0.9000** | **0.6618** | **0.7627** | **0.8296** | **0.9737** | **0.8328** |

### Confusion Matrix Breakdown (Test Set: 68 Failures, 1,932 Normal)
- **Decision Tree (Depth 3, 15 Nodes)**:
  - True Negatives ($\text{TN}$): $1,927$
  - False Positives ($\text{FP}$): $5$ (Only 5 false alarms)
  - False Negatives ($\text{FN}$): $50$
  - True Positives ($\text{TP}$): $18$
- **Random Forest**:
  - True Negatives ($\text{TN}$): $1,927$
  - False Positives ($\text{FP}$): $5$ (Only 5 false alarms)
  - False Negatives ($\text{FN}$): $23$
  - True Positives ($\text{TP}$): $45$
- **Baseline (Dummy)**:
  - $\text{TN} = 1932$, $\text{FP} = 0$, $\text{FN} = 68$ (Missed 100% of failures), $\text{TP} = 0$.

### Why False Negatives Matter in Predictive Maintenance
In factory operations, a **False Negative** means a machine failure goes undetected, leading to catastrophic tool fracture, motor burnout, and unscheduled production halts costing upwards of \$50,000. Conversely, a **False Positive** triggers a routine 15-minute technician inspection. Because the Decision Tree was tuned with `class_weight='balanced'`, it prioritizes high recall ($86.76\%$), catching $59$ of the $68$ test breakdowns.

---

## 7. Installation & Reproduction

Follow these commands to reproduce the lab from scratch:

```bash
# 1. Clone repository and navigate to root
cd predictive-maintenance-lab

# 2. Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# (For Linux / macOS)
# python3 -m venv .venv
# source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Download and persist official UCI dataset
python scripts/download_data.py

# 5. Train all models, perform hyperparameter optimization, and cache results
python -m src.train_models

# 6. Run unit test suite
python -m pytest tests/test_models.py -v

# 7. Launch the interactive Streamlit dashboard
streamlit run app.py
```

The Streamlit dashboard will be accessible locally at `http://localhost:8501`.

---

## 8. Project Structure

```
predictive-maintenance-lab/
├── app.py                     # Industrial control dashboard (5 modules)
├── requirements.txt           # Dependency specifications
├── README.md                  # System manual and documentation
├── project_report.md          # Academic report with methodology & results
├── data/
│   └── ai4i2020.csv           # Persisted UCI dataset
├── models/
│   ├── decision_tree.pkl      # Serialized Decision Tree Pipeline
│   ├── random_forest.pkl      # Serialized Random Forest Pipeline
│   ├── dummy_model.pkl        # Baseline Dummy Pipeline
│   └── evaluation_results.pkl # Metrics and curve data
├── src/
│   ├── data_loader.py         # Data loading and leakage filtering
│   ├── preprocessing.py      # ColumnTransformer & feature names extraction
│   ├── train_models.py        # GridSearchCV and RandomizedSearchCV
│   ├── evaluation.py          # Metric calculations & confusion matrices
│   ├── tree_explainer.py      # Decision path tracing & Plotly tree renderer
│   └── visualizations.py      # Plotly visual theme & UI widgets
├── scripts/
│   └── download_data.py       # UCI repo downloader
└── tests/
    └── test_models.py         # Pytest test suite
```

---

## 9. Known Limitations

1. **Synthetic Telemetry**: The AI4I 2020 dataset is a synthetic simulation generated to reflect industrial milling machines. Sensor distributions may have less stochastic noise than actual factory telemetry.
2. **Static Decision Trees vs. Drift**: The model is trained on stationary batch data. In production environments, sensor drift and seasonal ambient temperature shifts require continuous online re-training.
3. **Threshold Calibration**: The default $0.50$ probability threshold represents equal weighting between classes. In high-value manufacturing, operators should adjust the threshold via the UI slider to lower values (e.g. $0.25 - 0.35$) to further suppress False Negatives.

---

## 10. Academic References

- **Dataset**: Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
- **Decision Trees**: Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth & Brooks/Cole.
- **Random Forests**: Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- **Imbalanced Learning**: Saito, T., & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Imbalanced Datasets*. PLoS ONE, 10(3): e0118432.
- **Permutation Importance**: Fisher, A., Rudin, C., & Zhou, D. (2019). *All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Considering All Possible Prediction Models Simultaneously*. JMLR, 20(177), 1–81.
