"""
Predictive Maintenance Explainability Lab
Streamlit Industrial Analytics Dashboard & Decision Tree Explainer
Clean, aesthetic white theme with zero emojis.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_clean_data, FEATURE_COLS, TARGET_COL
from src.preprocessing import get_feature_names
from src.evaluation import evaluate_pipeline
from src.tree_explainer import (
    trace_sample_path,
    build_interactive_tree_figure,
    explain_gini_calculation,
    extract_node_data
)
from src.visualizations import (
    LIGHT_THEME,
    get_aesthetic_light_css,
    render_metric_card,
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_sensor_boxplots,
    plot_interactive_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_feature_importance_comparison,
    plot_train_vs_test_comparison,
    apply_light_theme
)

# -------------------------------------------------------------
# PAGE CONFIGURATION & CACHING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Explainability Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject clean white aesthetic CSS
st.markdown(get_aesthetic_light_css(), unsafe_allow_html=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


@st.cache_data
def get_dataset():
    """Load and cache verified dataset."""
    X, y, raw_df = load_clean_data()
    return X, y, raw_df


@st.cache_resource
def get_models_and_evaluation():
    """Load and cache trained models and evaluation artifacts."""
    dt_path = os.path.join(MODELS_DIR, "decision_tree.pkl")
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    dummy_path = os.path.join(MODELS_DIR, "dummy_model.pkl")
    eval_path = os.path.join(MODELS_DIR, "evaluation_results.pkl")

    if not all(os.path.exists(p) for p in [dt_path, rf_path, dummy_path, eval_path]):
        from src.train_models import run_training_pipeline
        run_training_pipeline()

    dt_pipeline = joblib.load(dt_path)
    rf_pipeline = joblib.load(rf_path)
    dummy_pipeline = joblib.load(dummy_path)
    eval_package = joblib.load(eval_path)

    return {
        "dt": dt_pipeline,
        "rf": rf_pipeline,
        "dummy": dummy_pipeline,
        "eval_package": eval_package
    }


# Load data and assets
X_data, y_data, raw_df = get_dataset()
assets = get_models_and_evaluation()
dt_model = assets["dt"]
rf_model = assets["rf"]
dummy_model = assets["dummy"]
eval_pkg = assets["eval_package"]
X_test = eval_pkg["test_data"]["X_test"]
y_test = eval_pkg["test_data"]["y_test"]

# Initialize session state for cross-tab sample communication
if "active_sample" not in st.session_state:
    st.session_state["active_sample"] = X_test.iloc[[0]].copy()
if "active_sample_origin" not in st.session_state:
    st.session_state["active_sample_origin"] = "Test Set Sample #0"

# -------------------------------------------------------------
# SIDEBAR NAVIGATION & GLOBAL CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### Industrial Analytics Lab")
    st.markdown(
        "<span style='color: #2563eb; font-weight: 600;'>Predictive Maintenance</span><br>"
        "<span style='color: #64748b; font-size: 0.85rem;'>Explainability and Tree Diagnostics</span>",
        unsafe_allow_html=True
    )
    st.divider()

    selected_section = st.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Data Explorer",
            "3. Model Lab",
            "4. Inside the Decision Tree",
            "5. Live Prediction"
        ],
        index=0
    )

    st.divider()
    st.markdown("#### Quick Dataset Facts")
    st.markdown(f"- **Observations:** 10,000")
    st.markdown(f"- **Input Sensors:** 6")
    st.markdown(f"- **Failure Rate:** 3.39% (Rare Event)")
    st.markdown(f"- **Target:** Machine failure (0 / 1)")

    st.divider()
    st.markdown(
        "<div style='font-size:0.75rem; color:#64748b; line-height: 1.5;'>"
        "Academic Machine Learning Lab project.<br>"
        "Dataset: UCI AI4I 2020 Predictive Maintenance (ID 601).<br>"
        "Fitted with Scikit-learn pipelines."
        "</div>",
        unsafe_allow_html=True
    )


# -------------------------------------------------------------
# 1. OVERVIEW SECTION
# -------------------------------------------------------------
if selected_section == "1. Overview":
    st.markdown("# Predictive Maintenance Explainability Lab")
    st.markdown(
        "#### *Industrial Sensor Fault Classification with White-Box Decision Tree Visual Diagnostics*"
    )
    st.markdown(
        "In modern manufacturing, unexpected machinery breakdown halts production lines and incurs substantial "
        "repair expenses. This academic lab demonstrates how multi-modal sensor telemetry can anticipate failure, "
        "and crucially, visualizes **how** a Decision Tree reasons about sensor measurements."
    )

    # Key Performance Metrics Row
    st.markdown("### Performance Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    rf_eval = eval_pkg["random_forest"]["eval"]
    dt_eval = eval_pkg["decision_tree"]["eval"]
    rf_metrics = rf_eval["metrics"]
    dt_metrics = dt_eval["metrics"]
    
    with col1:
        st.markdown(render_metric_card("Dataset Size", "10,000", "Synthetic telemetry samples", "#2563eb"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card("Class Imbalance", "3.39%", "339 failures / 9,661 normal", "#dc2626"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("Accuracy (RF)", f"{rf_metrics['Accuracy'] * 100:.1f}%", "Held-out test set", "#16a34a"), unsafe_allow_html=True)
    with col4:
        st.markdown(render_metric_card("Precision (RF)", f"{rf_metrics['Precision'] * 100:.1f}%", "High confidence alarms", "#4f46e5"), unsafe_allow_html=True)
    with col5:
        st.markdown(render_metric_card("ROC-AUC (RF)", f"{rf_metrics['ROC-AUC']:.3f}", "Discrimination power", "#0284c7"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architectural Overview & Lab Methodology
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### System Architecture & Methodology")
        st.markdown("""
        The system enforces strict academic machine learning standards:
        1. **Strict Feature Selection**: Exactly six predictors (`Type`, `Air temperature [K]`, `Process temperature [K]`, 
           `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`).
        2. **Data Leakage Shield**: Identifier columns (`UDI`, `Product ID`) and failure-type consequence columns 
           (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are quarantined to guarantee non-leaked, honest model performance.
        3. **Stratified Split**: 80:20 partition strictly preserving the 3.39% failure incidence across splits.
        4. **Class Imbalance Optimization**: Hyperparameter tuning optimizes **Average Precision** (`PR-AUC`) 
           over Stratified Cross-Validation rather than misleading raw accuracy.
        5. **Transparent Tree Structure**: The Decision Tree is trained with constrained depth so the tree 
           remains simple, clear, and directly understandable during examination.
        """)

    with col_right:
        st.markdown("### Top Sensor Predictors")
        rf_perm = eval_pkg["random_forest"]["importances"]["permutation"].head(5)
        fig_mini = px.bar(
            rf_perm,
            x="Importance_Mean",
            y="Feature",
            orientation="h",
            color="Importance_Mean",
            color_continuous_scale="Blues",
            labels={"Importance_Mean": "Permutation Importance (PR-AUC Loss)"}
        )
        fig_mini = apply_light_theme(fig_mini, "Most Critical Failure Predictors (Test Set)")
        fig_mini.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=270)
        st.plotly_chart(fig_mini, use_container_width=True)

    # Quick Jump Banner
    st.markdown(
        """
        <div class="aesthetic-notice">
            <b>Explore the Decision Traversal:</b> Navigate to <b>Section 4 (Inside the Decision Tree)</b> 
            to select any machine telemetry reading and trace its exact journey step-by-step through the learned tree structure.
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------------
# 2. DATA EXPLORER SECTION
# -------------------------------------------------------------
elif selected_section == "2. Data Explorer":
    st.markdown("# Exploratory Data Analysis & Integrity")
    st.markdown("Audit of sensor telemetry, missing values, correlation structures, and target class imbalance.")

    tab_preview, tab_dist, tab_corr, tab_box = st.tabs([
        "Dataset Preview & Schema",
        "Class Distribution & Imbalance",
        "Sensor Correlations",
        "Failure vs Normal Boxplots"
    ])

    with tab_preview:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(raw_df):,}")
        with col2:
            st.metric("Input Features", len(FEATURE_COLS))
        with col3:
            st.metric("Missing Values", int(raw_df[FEATURE_COLS].isnull().sum().sum()))
        with col4:
            st.metric("Duplicate Rows", int(raw_df[FEATURE_COLS].duplicated().sum()))

        st.markdown("#### Sample Records (Verified Predictors + Target)")
        preview_cols = FEATURE_COLS + [TARGET_COL]
        st.dataframe(raw_df[preview_cols].head(15), use_container_width=True)

        st.markdown("#### Quarantined Columns (Preventing Data Leakage)")
        st.markdown(
            "The raw AI4I dataset contains columns that must **not** be provided to the model: "
            "`UDI` and `Product ID` (arbitrary identifiers) and failure mode flags `TWF` (Tool Wear Failure), "
            "`HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), and `RNF` (Random Failure). "
            "These auxiliary labels directly reveal the target and would result in 100% artificial accuracy."
        )

    with tab_dist:
        col_imbalance, col_explanation = st.columns([1, 1])

        with col_imbalance:
            st.plotly_chart(plot_target_distribution(y_data), use_container_width=True)

        with col_explanation:
            st.markdown("### Why Accuracy is Misleading in Rare-Event Detection")
            st.markdown("""
            In predictive maintenance, failures are **rare events** (representing only **3.39%** of the dataset).

            - **The Majority-Class Baseline**: A model that simply predicts *'No Failure'* for every single machine achieves **96.61% accuracy**!
            - **The Operational Consequence**: That 96.61% accurate model catches **0 out of 68 failures** in the test set. In a factory, every missed failure leads to machine destruction and halted production lines.
            - **Evaluation Rule**: Never evaluate rare-event predictive maintenance using overall accuracy alone. Prioritize **Recall**, **Balanced Accuracy**, and **Average Precision (PR-AUC)**.
            """)

        st.divider()
        st.markdown("#### Numerical Feature Distributions")
        selected_numeric = st.selectbox(
            "Select sensor feature to inspect distribution:",
            ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        )
        fig_hist = px.histogram(
            raw_df,
            x=selected_numeric,
            color=TARGET_COL,
            marginal="box",
            barmode="overlay",
            color_discrete_map={0: "#10b981", 1: "#ef4444"},
            labels={str(TARGET_COL): "Machine Failure"}
        )
        fig_hist = apply_light_theme(fig_hist, f"Distribution of {selected_numeric} by Failure Status")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab_corr:
        st.markdown("### Sensor Correlation Analysis")
        st.markdown(
            "Notice the physical correlations: **Air Temperature** and **Process Temperature** exhibit strong positive correlation (~0.88), "
            "while **Rotational Speed** and **Torque** exhibit an inverse relationship."
        )
        num_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        st.plotly_chart(plot_correlation_heatmap(raw_df, num_cols), use_container_width=True)

    with tab_box:
        st.markdown("### Sensor Variance in Failure vs. Normal Operation")
        st.markdown("Compare the interquartile range (IQR) and outliers between operating states.")
        box_sensor = st.selectbox(
            "Select Sensor Measurement:",
            ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]", "Air temperature [K]", "Process temperature [K]"],
            key="box_sensor_selector"
        )
        st.plotly_chart(plot_sensor_boxplots(raw_df, box_sensor), use_container_width=True)


# -------------------------------------------------------------
# 3. MODEL LAB SECTION
# -------------------------------------------------------------
elif selected_section == "3. Model Lab":
    st.markdown("# Machine Learning Model Lab")
    st.markdown("Comparison across Baseline Dummy, Decision Tree, and Random Forest on untouched test data.")

    # Model Comparison Table
    st.markdown("### Test Set Model Benchmark")
    comp_df = eval_pkg["comparison_table"].copy()
    
    formatted_comp = comp_df.copy()
    for col in formatted_comp.columns:
        if col != "Model":
            formatted_comp[col] = formatted_comp[col].apply(lambda x: f"{x:.4f}")
    st.dataframe(formatted_comp, use_container_width=True)

    st.markdown(
        """
        <div class="aesthetic-notice">
            <b>Key Academic Takeaway:</b> The <b>Dummy Baseline</b> achieves <b>96.60% Accuracy</b> yet its <b>Recall, F1-Score, and Balanced Accuracy are 0.00%</b>! 
            The <b>Decision Tree</b> provides an interpretable structure with strong ROC-AUC (0.906), while the <b>Random Forest</b> delivers 90.00% precision and superior overall discrimination (ROC-AUC = 0.974).
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Interactive Probability Threshold Tuning
    st.markdown("### Operational Threshold Tuning")
    st.markdown(
        "In predictive maintenance, the standard 0.50 threshold can be tuned to balance false alarms against missed failures. "
        "Adjust the probability threshold below to observe real-time recalculation of the confusion matrix."
    )

    col_thresh_ctrl, col_model_ctrl = st.columns([2, 1])
    with col_thresh_ctrl:
        op_threshold = st.slider("Classification Probability Threshold:", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
    with col_model_ctrl:
        chosen_eval_model = st.selectbox("Select Model to Evaluate:", ["Decision Tree", "Random Forest"])

    active_pipeline = dt_model if chosen_eval_model == "Decision Tree" else rf_model
    dynamic_eval = evaluate_pipeline(active_pipeline, X_test, y_test, threshold=op_threshold)
    dyn_metrics = dynamic_eval["metrics"]
    dyn_cm = dynamic_eval["confusion_matrix"]

    # Dynamic KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("Operational Recall", f"{dyn_metrics['Recall'] * 100:.1f}%", f"Caught {dyn_cm['TP']} / {dyn_cm['TP'] + dyn_cm['FN']} failures", "#16a34a"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("Operational Precision", f"{dyn_metrics['Precision'] * 100:.1f}%", f"{dyn_cm['FP']} false alarms", "#2563eb"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("Missed Failures (FN)", f"{dyn_cm['FN']}", "Unpredicted failures", "#dc2626"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("Balanced Accuracy", f"{dyn_metrics['Balanced Accuracy'] * 100:.1f}%", "Macro average of classes", "#d97706"), unsafe_allow_html=True)

    col_cm_plot, col_cm_notes = st.columns([3, 2])
    with col_cm_plot:
        st.plotly_chart(plot_interactive_confusion_matrix(dyn_cm, f"{chosen_eval_model} (Threshold = {op_threshold:.2f})"), use_container_width=True)

    with col_cm_notes:
        st.markdown("#### The Operational Asymmetry of Errors")
        st.markdown(f"""
        - **False Negative (FN = {dyn_cm['FN']}):** The model predicts *Healthy*, but the machine breaks down. 
          Cost: Severe unplanned equipment damage, line stoppage, safety risk.
        - **False Positive (FP = {dyn_cm['FP']}):** The model predicts *Failure*, but the machine is fine. 
          Cost: A brief technician inspection and diagnostic test.
        - **Conclusion:** Industrial maintenance systems penalize False Negatives far more heavily than False Positives.
        """)

    st.divider()

    # ROC and PR Curves (Robust dictionary access)
    st.markdown("### Discrimination Curves")
    col_roc, col_pr = st.columns(2)
    models_dict = {
        "Baseline (Dummy)": eval_pkg["dummy"],
        "Decision Tree": eval_pkg["decision_tree"],
        "Random Forest": eval_pkg["random_forest"]
    }
    with col_roc:
        st.plotly_chart(plot_roc_curves(models_dict), use_container_width=True)
    with col_pr:
        st.plotly_chart(plot_pr_curves(models_dict), use_container_width=True)

    st.divider()

    # Feature Importance
    st.markdown("### Feature Importance Diagnostics: MDI vs. Permutation")
    st.markdown(
        "**Native Impurity Importance (MDI)** measures split impurity reductions during training. "
        "**Permutation Importance** measures performance loss when feature values are randomly shuffled on held-out test data, "
        "providing an unbiased assessment."
    )

    chosen_imp_model = st.selectbox("Inspect Feature Importance for:", ["Decision Tree", "Random Forest"], key="imp_model_selector")
    model_key = "decision_tree" if chosen_imp_model == "Decision Tree" else "random_forest"
    native_df = eval_pkg[model_key]["importances"]["native"]
    perm_df = eval_pkg[model_key]["importances"]["permutation"]
    st.plotly_chart(plot_feature_importance_comparison(native_df, perm_df, chosen_imp_model), use_container_width=True)

    st.divider()

    # Train vs Test (Overfitting Diagnosis)
    st.markdown("### Generalization Check (Train vs. Test)")
    train_vs_test_df = eval_pkg[model_key]["train_vs_test"]
    st.plotly_chart(plot_train_vs_test_comparison(train_vs_test_df, chosen_imp_model), use_container_width=True)


# -------------------------------------------------------------
# 4. INSIDE THE DECISION TREE (CORE FEATURE)
# -------------------------------------------------------------
elif selected_section == "4. Inside the Decision Tree":
    st.markdown("# Inside the Decision Tree: Understand How Your Model Thinks")
    st.markdown(
        "Trace the **exact root-to-leaf decision path** evaluated by the scikit-learn `DecisionTreeClassifier`. "
        "This simplified tree has depth 3 (15 total nodes), making each split and threshold straightforward to follow and explain."
    )

    # Sample Selection Options
    st.markdown("### 1. Select Telemetry Observation to Traverse")
    sample_source = st.radio(
        "Choose Telemetry Input Source:",
        ["Select Sample from Held-out Test Set", "Use Telemetry from Live Prediction Tab"],
        horizontal=True
    )

    if sample_source == "Select Sample from Held-out Test Set":
        col_filter, col_idx = st.columns([2, 3])
        with col_filter:
            filter_mode = st.selectbox("Filter Test Samples:", ["All Test Samples", "Actual Machine Failures (Target = 1)", "Actual Normal Machines (Target = 0)"])
        
        if filter_mode == "Actual Machine Failures (Target = 1)":
            filtered_indices = y_test[y_test == 1].index.tolist()
        elif filter_mode == "Actual Normal Machines (Target = 0)":
            filtered_indices = y_test[y_test == 0].index.tolist()
        else:
            filtered_indices = X_test.index.tolist()

        with col_idx:
            selected_idx = st.selectbox(
                f"Pick Observation Index (from {len(filtered_indices)} candidates):",
                filtered_indices,
                index=0
            )
            sample_df = X_test.loc[[selected_idx]].copy()
            actual_label = int(y_test.loc[selected_idx])
            st.session_state["active_sample"] = sample_df
            st.session_state["active_sample_origin"] = f"Test Sample #{selected_idx} (Actual Target: {'Failure' if actual_label == 1 else 'No Failure'})"
    else:
        sample_df = st.session_state["active_sample"]
        st.markdown(f"**Using Active Telemetry Input:** `{st.session_state['active_sample_origin']}`")

    # Display Current Sample Telemetry Card
    st.markdown("#### Input Sensor Telemetry")
    st.dataframe(sample_df, use_container_width=True)

    # Perform Root-to-Leaf Traversal using Scikit-Learn
    path_info = trace_sample_path(dt_model, sample_df)

    st.divider()

    # Traversal Stepper Cards
    st.markdown("### 2. Root-to-Leaf Prediction Path Walkthrough")
    st.markdown(
        "Follow each split decision node visited by this specific machine sample. "
        "Notice how the learned threshold is compared against the actual measured sensor value."
    )

    steps = path_info["steps"]
    for step in steps:
        eval_badge = "<span class='badge-true'>TRUE</span>" if step["evaluation"] else "<span class='badge-false'>FALSE</span>"
        branch_badge = f"<span class='badge-branch'>{step['branch']} BRANCH &rarr; Node #{step['next_node_id']}</span>"
        
        st.markdown(f"""
        <div class="step-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-weight:700; color:#2563eb; font-size:1.0rem;">STEP {step['step']}: Node #{step['node_id']}</span>
                <span>Branch Taken: {branch_badge}</span>
            </div>
            <div style="font-size:0.92rem; color:#0f172a; margin-bottom:4px;">
                <b>Condition Tested:</b> <code>{step['comparison']}</code> &nbsp;&rarr;&nbsp; Result: {eval_badge}
            </div>
            <div style="font-size:0.84rem; color:#64748b; line-height:1.4;">
                Sensor Reading: <code>{step['feature']} = {step['actual_value']:.2f}</code> | Learned Threshold: <code>&le; {step['threshold']:.2f}</code><br>
                Node Impurity: {step['gini']:.4f} | Training Samples: {step['n_samples']} | Impurity Reduction: {step['impurity_reduction']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Leaf Step Card
    leaf = path_info["leaf_data"]
    is_fail = (leaf["predicted_class"] == 1)
    card_class = "leaf-danger" if is_fail else "leaf-success"
    pred_color = "#dc2626" if is_fail else "#16a34a"

    st.markdown(f"""
    <div class="step-card {card_class}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-weight:700; color:{pred_color}; font-size:1.05rem;">FINAL LEAF: Node #{leaf['node_id']}</span>
            <span style="font-weight:700; color:{pred_color}; font-size:1.0rem; text-transform:uppercase;">
                CLASSIFICATION: {leaf['predicted_label']}
            </span>
        </div>
        <div style="font-size:0.88rem; color:#334155; line-height:1.5;">
            Leaf Impurity: {leaf['impurity']:.4f}<br>
            Training Samples in Leaf: {leaf['n_samples']} machines<br>
            Estimated Failure Probability: {path_info['probabilities']['Machine Failure'] * 100:.1f}%<br>
            Class Distribution: Normal: {leaf['class_counts'][0]}, Failure: {leaf['class_counts'][1]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Interactive Plotly Tree with Route Highlight
    st.markdown("### 3. Interactive Decision Tree Diagram")
    st.markdown(
        "- **Highlighted Blue Path:** The actual sequence of nodes and edges traversed by this sample.<br>"
        "- **Muted Nodes:** Unvisited branches for this specific machine reading.<br>"
        "- **Terminating Leaf:** The final decision node.",
        unsafe_allow_html=True
    )

    tree_fig = build_interactive_tree_figure(
        pipeline=dt_model,
        path_info=path_info,
        tree_title="Decision Tree (15 Nodes, Depth 3)"
    )
    tree_fig.update_layout(height=480)
    st.plotly_chart(tree_fig, use_container_width=True)

    st.divider()

    # Educational Gini Mathematics Section
    st.markdown("### 4. Educational Deep Dive: The Mathematics of Gini Impurity")
    st.markdown("""
    A Decision Tree chooses splits that maximize the **reduction in node impurity**. In classification, 
    **Gini Impurity** measures the probability that a randomly chosen element from the set would be incorrectly labeled 
    if it were randomly labeled according to the distribution of labels in the subset.
    """)

    st.latex(r"G = 1 - \sum_{i=1}^{C} p_i^2 = 1 - (p_0^2 + p_1^2)")

    st.markdown("""
    - $p_0$: Proportion of healthy machines in the node ($y = 0$).
    - $p_1$: Proportion of failed machines in the node ($y = 1$).
    - **Purity Limit ($G = 0.0$)**: All samples belong to a single class (maximum certainty).
    - **Maximum Impurity ($G = 0.5$)**: Equal 50/50 mix of failure and normal operations (maximum uncertainty).
    """)

    # Interactive Node Gini Calculator
    st.markdown("#### Calculate Gini Impurity for Any Traversed Node")
    step_choices = [f"Step {s['step']}: Node #{s['node_id']} ({s['feature']})" for s in steps]
    step_choices.append(f"Final Leaf #{leaf['node_id']} ({leaf['predicted_label']})")
    
    selected_step_choice = st.selectbox("Select Traversed Node to Inspect:", step_choices)
    
    if "Final Leaf" in selected_step_choice:
        chosen_node_data = leaf
    else:
        step_idx = int(selected_step_choice.split(":")[0].replace("Step ", "")) - 1
        chosen_node_data = extract_node_data(dt_model.named_steps["model"], get_feature_names(dt_model.named_steps["preprocessor"]))[steps[step_idx]["node_id"]]

    gini_calc = explain_gini_calculation(chosen_node_data)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Node Samples (N)", f"{gini_calc['n_samples']}")
    with c2:
        st.metric("Normal Ratio (p0)", f"{gini_calc['p0']:.4f}")
    with c3:
        st.metric("Failure Ratio (p1)", f"{gini_calc['p1']:.4f}")
    with c4:
        st.metric("Node Gini Impurity", f"{gini_calc['gini']:.4f}")

    st.markdown(f"""
    **Calculation Breakdown for Node #{chosen_node_data['node_id']}:**
    - $p_0^2 = ({gini_calc['p0']:.4f})^2 = {gini_calc['p0_squared']:.4f}$
    - $p_1^2 = ({gini_calc['p1']:.4f})^2 = {gini_calc['p1_squared']:.4f}$
    - $\\sum p_i^2 = {gini_calc['sum_squared']:.4f}$
    - $G = 1 - {gini_calc['sum_squared']:.4f} = \\mathbf{{{gini_calc['gini']:.4f}}}$
    """)

    if not chosen_node_data["is_leaf"]:
        st.markdown(f"""
        **Impurity Reduction of this Split (\\Delta G):**
        $$\\Delta G = G_{{parent}} - \\left( \\frac{{N_{{left}}}}{{N_{{parent}}}} G_{{left}} + \\frac{{N_{{right}}}}{{N_{{parent}}}} G_{{right}} \\right) = \\mathbf{{{chosen_node_data['impurity_reduction']:.4f}}}$$
        The decision tree selects the split condition that yields the greatest \\Delta G among all candidate features.
        """)


# -------------------------------------------------------------
# 5. LIVE PREDICTION SECTION
# -------------------------------------------------------------
elif selected_section == "5. Live Prediction":
    st.markdown("# Live Machine Failure Prediction Interface")
    st.markdown(
        "Enter industrial telemetry parameters manually or choose an operational scenario preset to test real-time inference."
    )

    # Preset Scenarios
    st.markdown("### Operational Scenario Presets")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    preset_values = None
    with col_p1:
        if st.button("Normal Healthy Operation", use_container_width=True):
            preset_values = {"Type": "M", "Air": 298.1, "Proc": 308.6, "Speed": 1550, "Torque": 42.5, "Wear": 45}
    with col_p2:
        if st.button("High Tool Wear Warning", use_container_width=True):
            preset_values = {"Type": "L", "Air": 300.2, "Proc": 310.5, "Speed": 1380, "Torque": 58.0, "Wear": 235}
    with col_p3:
        if st.button("Over-Torque Stall Anomaly", use_container_width=True):
            preset_values = {"Type": "H", "Air": 302.5, "Proc": 311.8, "Speed": 1150, "Torque": 72.0, "Wear": 160}
    with col_p4:
        if st.button("Heat Dissipation Issue", use_container_width=True):
            preset_values = {"Type": "L", "Air": 304.2, "Proc": 312.9, "Speed": 1320, "Torque": 55.0, "Wear": 190}

    # Sensor Inputs Form
    with st.form("prediction_form"):
        st.markdown("### Sensor Telemetry Inputs")
        st.caption("Realistic bounds derived from the UCI AI4I 2020 dataset.")

        c1, c2, c3 = st.columns(3)
        with c1:
            val_type = st.selectbox(
                "Product Quality Variant (Type):",
                ["L", "M", "H"],
                index=["L", "M", "H"].index(preset_values["Type"]) if preset_values else 1,
                help="L = Low quality (50%), M = Medium quality (30%), H = High quality (20%)"
            )
            val_air = st.slider(
                "Air Temperature [K]:",
                min_value=295.0, max_value=305.0,
                value=preset_values["Air"] if preset_values else 298.5,
                step=0.1
            )
        with c2:
            val_proc = st.slider(
                "Process Temperature [K]:",
                min_value=305.0, max_value=315.0,
                value=preset_values["Proc"] if preset_values else 308.7,
                step=0.1
            )
            val_speed = st.slider(
                "Rotational Speed [rpm]:",
                min_value=1100, max_value=2900,
                value=preset_values["Speed"] if preset_values else 1500,
                step=10
            )
        with c3:
            val_torque = st.slider(
                "Torque [Nm]:",
                min_value=3.0, max_value=80.0,
                value=preset_values["Torque"] if preset_values else 40.0,
                step=0.5
            )
            val_wear = st.slider(
                "Tool Wear [min]:",
                min_value=0, max_value=260,
                value=preset_values["Wear"] if preset_values else 80,
                step=1
            )

        st.markdown("### Inference Configuration")
        c_model, c_thresh = st.columns([2, 2])
        with c_model:
            selected_model_name = st.radio("Active Classifier:", ["Decision Tree", "Random Forest"], horizontal=True)
        with c_thresh:
            pred_threshold = st.slider("Alarm Decision Threshold:", min_value=0.05, max_value=0.95, value=0.50, step=0.05)

        submit_btn = st.form_submit_button("Predict Machine Operational Status", use_container_width=True)

    # Process Prediction
    input_df = pd.DataFrame([{
        "Type": val_type,
        "Air temperature [K]": float(val_air),
        "Process temperature [K]": float(val_proc),
        "Rotational speed [rpm]": int(val_speed),
        "Torque [Nm]": float(val_torque),
        "Tool wear [min]": int(val_wear)
    }])

    # Update active sample for Tree Explainer
    st.session_state["active_sample"] = input_df
    st.session_state["active_sample_origin"] = f"Live Input (Type={val_type}, Torque={val_torque}Nm, Speed={val_speed}rpm, Wear={val_wear}min)"

    active_pipe = dt_model if selected_model_name == "Decision Tree" else rf_model
    probas = active_pipe.predict_proba(input_df)[0]
    p_fail = float(probas[1])
    is_failure = p_fail >= pred_threshold

    st.markdown("### Inference Diagnostic Output")
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        status_label = "MACHINE FAILURE PREDICTED" if is_failure else "NORMAL OPERATION"
        status_color = "#dc2626" if is_failure else "#16a34a"
        st.markdown(render_metric_card("Diagnostic Status", status_label, f"Threshold = {pred_threshold:.2f}", status_color), unsafe_allow_html=True)

    with col_res2:
        st.markdown(render_metric_card("Failure Probability", f"{p_fail * 100:.1f}%", f"Healthy probability: {(1 - p_fail) * 100:.1f}%", "#2563eb"), unsafe_allow_html=True)

    with col_res3:
        risk_level = "CRITICAL RISK" if p_fail > 0.65 else ("MODERATE WARNING" if p_fail > 0.30 else "LOW / SAFE")
        risk_color = "#dc2626" if p_fail > 0.65 else ("#d97706" if p_fail > 0.30 else "#16a34a")
        st.markdown(render_metric_card("Operational Risk", risk_level, f"Model: {selected_model_name}", risk_color), unsafe_allow_html=True)

    # Visual Probability Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_fail * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Failure Risk Gauge (%)", 'font': {'size': 14, 'color': '#0f172a'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': status_color},
            'bgcolor': "#ffffff",
            'borderwidth': 1,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': '#dcfce7'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#d97706", 'width': 3},
                'thickness': 0.75,
                'value': pred_threshold * 100
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
        height=240,
        margin=dict(l=30, r=30, t=40, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Bridge to Tree Explainer
    st.markdown("### Explain This Exact Prediction in the Decision Tree")
    st.markdown(
        "Want to see which specific sensor rule triggered this prediction? "
        "Click the button below to load these exact sensor measurements into the **Inside the Decision Tree** module."
    )
    if st.button("Open in Inside the Decision Tree Module", use_container_width=True):
        st.success("Sample telemetry successfully linked! Switch to Section 4 ('Inside the Decision Tree') in the sidebar.")

    st.markdown(
        """
        <div class="aesthetic-notice" style="border-left-color: #2563eb;">
            <b>Academic Disclaimer:</b> This machine learning application is built for educational and demonstrative 
            purposes using the synthetic UCI AI4I 2020 Predictive Maintenance dataset. It is strictly not intended for 
            direct control or life-safety decisions in actual heavy industrial manufacturing plants.
        </div>
        """,
        unsafe_allow_html=True
    )
