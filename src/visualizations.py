"""
Visualization utilities for the Predictive Maintenance Explainability Lab.
Provides clean, aesthetic white-themed Plotly charts, confusion matrix heatmaps,
ROC/PR curves, feature importance plots, and modern dashboard styling.
Completely emoji-free and optimized for high visual readability.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Aesthetic Light Theme Palette
LIGHT_THEME = {
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff",
    "card_bg": "#ffffff",
    "border_color": "#e2e8f0",
    "text_primary": "#0f172a",
    "text_muted": "#64748b",
    "text_dark": "#1e293b",
    "accent_blue": "#2563eb",
    "accent_indigo": "#4f46e5",
    "accent_green": "#16a34a",
    "accent_red": "#dc2626",
    "accent_amber": "#d97706",
    "grid_color": "#f1f5f9"
}


def apply_light_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies standardized sleek, aesthetic light theme to any Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color=LIGHT_THEME["text_primary"], family="Inter, system-ui, sans-serif")
        ),
        paper_bgcolor=LIGHT_THEME["paper_bgcolor"],
        plot_bgcolor=LIGHT_THEME["plot_bgcolor"],
        font=dict(color=LIGHT_THEME["text_muted"], family="Inter, system-ui, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=LIGHT_THEME["grid_color"],
            zerolinecolor="#cbd5e1",
            tickfont=dict(color=LIGHT_THEME["text_muted"], size=11),
            title_font=dict(color=LIGHT_THEME["text_dark"], size=12)
        ),
        yaxis=dict(
            gridcolor=LIGHT_THEME["grid_color"],
            zerolinecolor="#cbd5e1",
            tickfont=dict(color=LIGHT_THEME["text_muted"], size=11),
            title_font=dict(color=LIGHT_THEME["text_dark"], size=12)
        ),
        legend=dict(
            font=dict(color=LIGHT_THEME["text_dark"], size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=LIGHT_THEME["border_color"],
            borderwidth=1
        )
    )
    return fig


# -------------------------------------------------------------
# EDA VISUALIZATIONS
# -------------------------------------------------------------

def plot_target_distribution(y: pd.Series) -> go.Figure:
    """Clean donut chart illustrating class distribution."""
    counts = y.value_counts().rename({0: "Normal (0)", 1: "Machine Failure (1)"})
    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.62,
        marker=dict(
            colors=["#10b981", "#ef4444"],
            line=dict(color="#ffffff", width=2)
        ),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(color="#1e293b", size=12),
        hovertemplate="<b>%{label}</b><br>Observations: %{value:,}<br>Proportion: %{percent}<extra></extra>"
    )])
    fig.update_layout(
        title=dict(text="Machine Failure Class Distribution", font=dict(size=14, color=LIGHT_THEME["text_primary"])),
        paper_bgcolor=LIGHT_THEME["paper_bgcolor"],
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=False,
        height=320
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
    """Plotly interactive heatmap showing Pearson correlation among sensor measurements."""
    corr = df[numeric_cols].corr()
    
    # Custom text colors for legibility against heatmap cells
    z_vals = corr.values
    text_colors = []
    for row in z_vals:
        row_colors = []
        for val in row:
            row_colors.append("#ffffff" if abs(val) > 0.6 else "#0f172a")
        text_colors.append(row_colors)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=corr.columns,
        y=corr.columns,
        colorscale="Blues",
        zmin=-1,
        zmax=1,
        text=np.round(z_vals, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="#0f172a"),
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color=LIGHT_THEME["text_dark"], size=11)),
            tickfont=dict(color=LIGHT_THEME["text_muted"], size=10)
        )
    ))
    fig.update_layout(
        title=dict(text="Sensor Telemetry Correlation Matrix", font=dict(size=14, color=LIGHT_THEME["text_primary"])),
        paper_bgcolor=LIGHT_THEME["paper_bgcolor"],
        plot_bgcolor=LIGHT_THEME["paper_bgcolor"],
        margin=dict(l=60, r=40, t=50, b=50),
        font=dict(color=LIGHT_THEME["text_dark"])
    )
    return fig


def plot_sensor_boxplots(df: pd.DataFrame, sensor_col: str, target_col: str = "Machine failure") -> go.Figure:
    """Boxplot comparing sensor distribution for normal vs failed machines."""
    df_plot = df.copy()
    df_plot["Status"] = df_plot[target_col].map({0: "Normal Operation", 1: "Machine Failure"})
    
    fig = px.box(
        df_plot,
        x="Status",
        y=sensor_col,
        color="Status",
        color_discrete_map={"Normal Operation": "#10b981", "Machine Failure": "#ef4444"}
    )
    fig = apply_light_theme(fig, f"Sensor Distribution: {sensor_col}")
    fig.update_layout(showlegend=False, height=360)
    return fig


# -------------------------------------------------------------
# MODEL LAB VISUALIZATIONS
# -------------------------------------------------------------

def plot_interactive_confusion_matrix(cm_dict: Dict[str, Any], model_name: str) -> go.Figure:
    """Renders confusion matrix with perfectly legible contrasting text in every cell."""
    cm = cm_dict["Matrix"]
    tn, fp, fn, tp = cm_dict["TN"], cm_dict["FP"], cm_dict["FN"], cm_dict["TP"]

    z = [[tn, fp], [fn, tp]]
    
    # Custom annotations with explicit color contrast
    # TN is largest (dark blue cell -> white text)
    # FP, FN, TP are smaller (light cells -> dark slate text)
    annotations = [
        go.layout.Annotation(
            x=0, y=0,
            text=f"<b>True Negative (TN)</b><br><b>{tn:,}</b><br><span style='font-size:10px;'>Correct Normal</span>",
            showarrow=False,
            font=dict(color="#ffffff", size=13)
        ),
        go.layout.Annotation(
            x=1, y=0,
            text=f"<b>False Positive (FP)</b><br><b>{fp:,}</b><br><span style='font-size:10px;'>False Alarm</span>",
            showarrow=False,
            font=dict(color="#0f172a", size=13)
        ),
        go.layout.Annotation(
            x=0, y=1,
            text=f"<b>False Negative (FN)</b><br><b>{fn:,}</b><br><span style='font-size:10px;'>Missed Failure</span>",
            showarrow=False,
            font=dict(color="#0f172a", size=13)
        ),
        go.layout.Annotation(
            x=1, y=1,
            text=f"<b>True Positive (TP)</b><br><b>{tp:,}</b><br><span style='font-size:10px;'>Failure Caught</span>",
            showarrow=False,
            font=dict(color="#0f172a" if tp < 500 else "#ffffff", size=13)
        ),
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted: Normal (0)", "Predicted: Failure (1)"],
        y=["Actual: Normal (0)", "Actual: Failure (1)"],
        colorscale=[
            [0.0, "#f8fafc"],
            [0.1, "#dbeafe"],
            [0.3, "#93c5fd"],
            [0.7, "#2563eb"],
            [1.0, "#1e3a8a"]
        ],
        showscale=False
    ))

    fig.update_layout(
        annotations=annotations,
        title=dict(text=f"Confusion Matrix: {model_name}", font=dict(size=14, color=LIGHT_THEME["text_primary"])),
        paper_bgcolor=LIGHT_THEME["paper_bgcolor"],
        plot_bgcolor=LIGHT_THEME["paper_bgcolor"],
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(tickfont=dict(color=LIGHT_THEME["text_dark"], size=11)),
        yaxis=dict(tickfont=dict(color=LIGHT_THEME["text_dark"], size=11), autorange="reversed"),
        height=340
    )
    return fig


def plot_roc_curves(models_eval: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plots combined ROC curves for all models with AUC scores. Completely resilient to eval nesting."""
    fig = go.Figure()

    colors = {
        "Baseline (Dummy)": "#94a3b8",
        "Decision Tree": "#2563eb",
        "Random Forest": "#059669"
    }

    # Reference diagonal (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#cbd5e1", dash="dash", width=1.5),
        name="Chance (AUC = 0.50)"
    ))

    for name, data in models_eval.items():
        data_eval = data.get("eval", data)
        roc = data_eval["roc_curve"]
        auc = data_eval["metrics"]["ROC-AUC"]
        fig.add_trace(go.Scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            line=dict(color=colors.get(name, "#2563eb"), width=2.5),
            name=f"{name} (AUC = {auc:.3f})"
        ))

    fig = apply_light_theme(fig, "Receiver Operating Characteristic (ROC) Comparison")
    fig.update_xaxes(title="False Positive Rate (1 - Specificity)")
    fig.update_yaxes(title="True Positive Rate (Recall)")
    fig.update_layout(height=360)
    return fig


def plot_pr_curves(models_eval: Dict[str, Dict[str, Any]], baseline_prev: float = 0.034) -> go.Figure:
    """Plots Precision-Recall curves highlighting average precision on rare failure class."""
    fig = go.Figure()

    colors = {
        "Baseline (Dummy)": "#94a3b8",
        "Decision Tree": "#2563eb",
        "Random Forest": "#059669"
    }

    # Baseline prevalence horizontal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline_prev, baseline_prev],
        mode="lines",
        line=dict(color="#cbd5e1", dash="dash", width=1.5),
        name=f"No-Skill Baseline ({baseline_prev * 100:.1f}%)"
    ))

    for name, data in models_eval.items():
        data_eval = data.get("eval", data)
        pr = data_eval["pr_curve"]
        ap = data_eval["metrics"]["Average Precision"]
        fig.add_trace(go.Scatter(
            x=pr["recall"],
            y=pr["precision"],
            mode="lines",
            line=dict(color=colors.get(name, "#2563eb"), width=2.5),
            name=f"{name} (Avg Prec = {ap:.3f})"
        ))

    fig = apply_light_theme(fig, "Precision-Recall (PR) Curves")
    fig.update_xaxes(title="Recall (Failures Detected)")
    fig.update_yaxes(title="Precision (Alarm Accuracy)")
    fig.update_layout(height=360)
    return fig


def plot_feature_importance_comparison(
    native_df: Optional[pd.DataFrame],
    perm_df: Optional[pd.DataFrame],
    model_name: str
) -> go.Figure:
    """Plots clean horizontal bar chart of feature importances."""
    fig = go.Figure()

    if native_df is not None:
        fig.add_trace(go.Bar(
            y=native_df["Feature"],
            x=native_df["Importance"],
            orientation="h",
            name="Native Impurity (MDI)",
            marker=dict(color="#93c5fd")
        ))

    if perm_df is not None:
        fig.add_trace(go.Bar(
            y=perm_df["Feature"],
            x=perm_df["Importance_Mean"],
            error_x=dict(type="data", array=perm_df["Importance_Std"], color="#1e3a8a"),
            orientation="h",
            name="Permutation Importance (Test Set)",
            marker=dict(color="#2563eb")
        ))

    fig = apply_light_theme(fig, f"Feature Importance: {model_name}")
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"), height=360)
    fig.update_xaxes(title="Importance Score")
    return fig


def plot_train_vs_test_comparison(train_vs_test_df: pd.DataFrame, model_name: str) -> go.Figure:
    """Bar chart comparing Train vs Test metrics to visualize generalization gap."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=train_vs_test_df["Metric"],
        y=train_vs_test_df["Train"],
        name="Training Set",
        marker=dict(color="#93c5fd")
    ))

    fig.add_trace(go.Bar(
        x=train_vs_test_df["Metric"],
        y=train_vs_test_df["Test"],
        name="Held-out Test Set",
        marker=dict(color="#2563eb")
    ))

    fig = apply_light_theme(fig, f"Train vs. Test Performance: {model_name}")
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.05]), height=360)
    return fig


# -------------------------------------------------------------
# CSS AND HTML CARD HELPERS (AESTHETIC LIGHT THEME)
# -------------------------------------------------------------

def get_aesthetic_light_css() -> str:
    """Custom CSS for Streamlit providing a modern, clean, aesthetic white/light theme."""
    return """
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Ensure all text elements are dark and crystal clear */
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #0f172a;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    /* Form labels, widget labels, radio and slider labels */
    div[data-testid="stWidgetLabel"] label, 
    div[data-testid="stWidgetLabel"] p,
    div[data-baseweb="radio"] div,
    div[data-baseweb="radio"] label,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span {
        color: #0f172a !important;
        font-weight: 500 !important;
    }

    /* Streamlit preset & secondary buttons - clean white with border and dark text */
    .stButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #f8fafc !important;
        border-color: #2563eb !important;
        color: #2563eb !important;
    }

    /* Primary form submit button - vibrant royal blue with white text */
    .stFormSubmitButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        box-shadow: 0 1px 3px 0 rgba(37, 99, 235, 0.25) !important;
    }
    .stFormSubmitButton > button:hover {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }

    /* Selectbox and dropdown styling */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
    }
    div[data-baseweb="select"] span {
        color: #0f172a !important;
    }

    /* Slider values and track */
    div[data-testid="stSlider"] div {
        color: #0f172a !important;
    }

    /* Dataframe container */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        background-color: #ffffff;
    }

    /* Metrics Card Container */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Traversal Step Card */
    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }
    .step-card.leaf-success {
        border-left: 4px solid #16a34a;
        background: #f0fdf4;
    }
    .step-card.leaf-danger {
        border-left: 4px solid #dc2626;
        background: #fef2f2;
    }
    
    .badge-true {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-false {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .badge-branch {
        background-color: #dbeafe;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }

    /* Warning / Notice Banner */
    .aesthetic-notice {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #334155;
        margin: 14px 0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    </style>
    """


def render_metric_card(label: str, value: str, sub: str = "", border_color: str = "#2563eb") -> str:
    """Returns HTML for a clean white-themed metric card."""
    return f"""
    <div class="metric-card" style="border-top: 3px solid {border_color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{sub}</div>' if sub else ''}
    </div>
    """
