"""
Tree Explainer module for scikit-learn DecisionTreeClassifier.
Extracts exact decision paths, calculates Gini reductions, maps one-hot feature names,
and generates an interactive, route-highlighted Plotly tree diagram.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

from src.preprocessing import get_feature_names


def extract_node_data(tree_model, feature_names: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    Extracts complete structural metadata for all nodes in the DecisionTree.
    """
    tree_ = tree_model.tree_
    n_nodes = tree_.node_count
    nodes_info = {}

    for node_id in range(n_nodes):
        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]
        feature_idx = tree_.feature[node_id]
        threshold = tree_.threshold[node_id]
        impurity = float(tree_.impurity[node_id])
        n_samples = int(tree_.n_node_samples[node_id])
        values = tree_.value[node_id][0]
        total_val = float(values.sum())
        p0 = float(values[0] / total_val) if total_val > 0 else 0.5
        p1 = float(values[1] / total_val) if total_val > 0 else 0.5
        predicted_class = int(np.argmax(values))

        feature_name = feature_names[feature_idx] if not is_leaf else ""

        left_child = int(tree_.children_left[node_id])
        right_child = int(tree_.children_right[node_id])

        # Impurity reduction calculation if internal node
        impurity_reduction = 0.0
        if not is_leaf and left_child != -1 and right_child != -1:
            n_left = tree_.n_node_samples[left_child]
            n_right = tree_.n_node_samples[right_child]
            gini_left = tree_.impurity[left_child]
            gini_right = tree_.impurity[right_child]
            weighted_child_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
            impurity_reduction = float(impurity - weighted_child_gini)

        # Calculate estimated sample distribution counts
        c0 = int(round(p0 * n_samples))
        c1 = int(n_samples - c0)

        nodes_info[node_id] = {
            "node_id": node_id,
            "is_leaf": is_leaf,
            "feature_idx": feature_idx,
            "feature_name": feature_name,
            "threshold": float(threshold) if not is_leaf else None,
            "impurity": impurity,
            "n_samples": n_samples,
            "class_counts": [c0, c1],
            "weighted_values": [float(values[0]), float(values[1])],
            "proportions": [p0, p1],
            "predicted_class": predicted_class,
            "predicted_label": "Machine Failure" if predicted_class == 1 else "No Failure",
            "left_child": left_child,
            "right_child": right_child,
            "impurity_reduction": impurity_reduction
        }

    return nodes_info


def compute_tree_coordinates(tree_model) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Computes (x, y) coordinates for binary tree visualization.
    Uses bottom-up midpoint layout for internal nodes to ensure optimal horizontal spacing.
    """
    tree_ = tree_model.tree_

    # Determine depth of each node
    depths = {}
    def calc_depth(node_id: int, current_depth: int):
        depths[node_id] = current_depth
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left != -1:
            calc_depth(left, current_depth + 1)
        if right != -1:
            calc_depth(right, current_depth + 1)

    calc_depth(0, 0)

    # Assign x-coordinates: leaves ordered sequentially, parents centered over children
    leaf_counter = [0.0]
    x_coords = {}

    def assign_x(node_id: int):
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        is_leaf = (left == right)

        if is_leaf:
            x_coords[node_id] = leaf_counter[0]
            leaf_counter[0] += 1.0
        else:
            assign_x(left)
            assign_x(right)
            x_coords[node_id] = (x_coords[left] + x_coords[right]) / 2.0

    assign_x(0)

    # Invert depth so root is at top
    max_d = max(depths.values()) if depths else 1
    y_coords = {nid: float(max_d - d) for nid, d in depths.items()}

    return x_coords, y_coords


def trace_sample_path(
    pipeline: Pipeline,
    sample_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Traverses the actual fitted DecisionTreeClassifier for a single sample.
    Uses decision_path() and apply() from scikit-learn.
    Returns step-by-step condition evaluations, boolean results, and chosen branches.
    """
    if len(sample_df) != 1:
        raise ValueError("trace_sample_path requires a DataFrame with exactly 1 row.")

    preprocessor = pipeline.named_steps["preprocessor"]
    tree_model = pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocessor)

    # Transform raw sample using exact fitted preprocessing pipeline
    X_trans = preprocessor.transform(sample_df)
    transformed_features = pd.Series(X_trans[0], index=feature_names)

    # Scikit-learn decision path & leaf prediction
    indicator = tree_model.decision_path(X_trans)
    visited_node_ids = list(indicator.indices)
    leaf_id = int(tree_model.apply(X_trans)[0])

    nodes_info = extract_node_data(tree_model, feature_names)

    steps = []
    edges_traversed = []

    for i in range(len(visited_node_ids) - 1):
        current_node_id = visited_node_ids[i]
        next_node_id = visited_node_ids[i + 1]
        node_data = nodes_info[current_node_id]

        feat_name = node_data["feature_name"]
        thresh = node_data["threshold"]
        actual_val = float(transformed_features[feat_name])

        is_left = (next_node_id == node_data["left_child"])
        condition_eval = (actual_val <= thresh)
        branch = "LEFT" if is_left else "RIGHT"

        edges_traversed.append((current_node_id, next_node_id))

        # Impurity reduction at this step
        imp_red = node_data["impurity_reduction"]

        steps.append({
            "step": i + 1,
            "node_id": current_node_id,
            "feature": feat_name,
            "threshold": thresh,
            "actual_value": actual_val,
            "comparison": f"{feat_name} ({actual_val:.2f}) <= {thresh:.2f}",
            "evaluation": condition_eval,
            "branch": branch,
            "next_node_id": next_node_id,
            "gini": node_data["impurity"],
            "n_samples": node_data["n_samples"],
            "class_counts": node_data["class_counts"],
            "impurity_reduction": imp_red
        })

    leaf_data = nodes_info[leaf_id]
    probas = pipeline.predict_proba(sample_df)[0]
    final_pred = int(pipeline.predict(sample_df)[0])

    return {
        "visited_nodes": visited_node_ids,
        "leaf_id": leaf_id,
        "edges_traversed": edges_traversed,
        "steps": steps,
        "leaf_data": leaf_data,
        "probabilities": {"No Failure": float(probas[0]), "Machine Failure": float(probas[1])},
        "final_prediction": final_pred,
        "final_prediction_label": "Machine Failure" if final_pred == 1 else "No Failure",
        "sample_values": sample_df.iloc[0].to_dict(),
        "transformed_values": transformed_features.to_dict()
    }


def build_interactive_tree_figure(
    pipeline: Pipeline,
    path_info: Optional[Dict[str, Any]] = None,
    tree_title: str = "Decision Tree Architecture & Traversal"
) -> go.Figure:
    """
    Renders the fitted Decision Tree as an interactive, fully responsive Plotly figure.
    Dynamically highlights visited nodes and edges when path_info is provided.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    tree_model = pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocessor)

    nodes_info = extract_node_data(tree_model, feature_names)
    x_coords, y_coords = compute_tree_coordinates(tree_model)

    visited_nodes = set(path_info["visited_nodes"]) if path_info else set()
    final_leaf = path_info["leaf_id"] if path_info else None
    visited_edges = set(path_info["edges_traversed"]) if path_info else set()

    fig = go.Figure()

    # 1. DRAW EDGES
    for node_id, data in nodes_info.items():
        if not data["is_leaf"]:
            for child_id in [data["left_child"], data["right_child"]]:
                is_traversed = (node_id, child_id) in visited_edges
                
                # Visual hierarchy: highlighted edges vs muted edges
                if path_info:
                    line_color = "#2563eb" if is_traversed else "rgba(203, 213, 225, 0.5)"
                    line_width = 4.0 if is_traversed else 1.2
                else:
                    line_color = "#94a3b8"
                    line_width = 1.6

                fig.add_trace(go.Scatter(
                    x=[x_coords[node_id], x_coords[child_id]],
                    y=[y_coords[node_id], y_coords[child_id]],
                    mode="lines",
                    line=dict(color=line_color, width=line_width),
                    hoverinfo="none",
                    showlegend=False
                ))

    # 2. DRAW NODES
    # Clean white theme node styling
    categories = [
        {"name": "Decision Node", "color": "#f8fafc", "border": "#2563eb", "text_color": "#0f172a", "shape": "circle"},
        {"name": "Normal Operation Leaf", "color": "#f0fdf4", "border": "#16a34a", "text_color": "#166534", "shape": "square"},
        {"name": "Machine Failure Leaf", "color": "#fef2f2", "border": "#dc2626", "text_color": "#991b1b", "shape": "diamond"},
    ]

    for cat in categories:
        xs, ys, texts, hovers, sizes, line_colors, line_widths, opacities, text_colors = [], [], [], [], [], [], [], [], []

        for node_id, data in nodes_info.items():
            # Category match check
            if not data["is_leaf"]:
                if cat["name"] != "Decision Node":
                    continue
            else:
                if data["predicted_class"] == 0 and cat["name"] != "Normal Operation Leaf":
                    continue
                if data["predicted_class"] == 1 and cat["name"] != "Machine Failure Leaf":
                    continue

            x = x_coords[node_id]
            y = y_coords[node_id]

            # Hover text details
            if not data["is_leaf"]:
                hover = (
                    f"<b>Node #{node_id} (Split)</b><br>"
                    f"<b>Feature:</b> {data['feature_name']}<br>"
                    f"<b>Threshold:</b> &le; {data['threshold']:.2f}<br>"
                    f"<b>Impurity:</b> {data['impurity']:.4f}<br>"
                    f"<b>Samples:</b> {data['n_samples']}<br>"
                    f"<b>Distribution:</b> [Normal: {data['class_counts'][0]}, Failure: {data['class_counts'][1]}]<br>"
                    f"<b>Impurity Gain &Delta;G:</b> {data['impurity_reduction']:.4f}"
                )
            else:
                hover = (
                    f"<b>Leaf #{node_id} ({data['predicted_label']})</b><br>"
                    f"<b>Impurity:</b> {data['impurity']:.4f}<br>"
                    f"<b>Samples:</b> {data['n_samples']}<br>"
                    f"<b>Failure Ratio:</b> {data['proportions'][1] * 100:.1f}%<br>"
                    f"<b>Distribution:</b> [Normal: {data['class_counts'][0]}, Failure: {data['class_counts'][1]}]"
                )

            # Highlighting logic for sample journey
            is_visited = node_id in visited_nodes
            is_final = (node_id == final_leaf)

            if path_info:
                if is_final:
                    node_size = 38
                    l_color = "#d97706"  # Warm amber border
                    l_width = 4.0
                    opacity = 1.0
                elif is_visited:
                    node_size = 32
                    l_color = "#2563eb"  # Royal blue
                    l_width = 3.5
                    opacity = 1.0
                else:
                    node_size = 22
                    l_color = cat["border"]
                    l_width = 1.0
                    opacity = 0.30
            else:
                node_size = 28
                l_color = cat["border"]
                l_width = 1.8
                opacity = 1.0

            xs.append(x)
            ys.append(y)
            texts.append(f"#{node_id}")
            hovers.append(hover)
            sizes.append(node_size)
            line_colors.append(l_color)
            line_widths.append(l_width)
            opacities.append(opacity)
            text_colors.append(cat["text_color"])

        if xs:
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=cat["name"],
                marker=dict(
                    symbol=cat["shape"],
                    size=sizes,
                    color=cat["color"],
                    line=dict(color=line_colors, width=line_widths),
                    opacity=opacities
                ),
                text=texts,
                textposition="middle center",
                textfont=dict(size=11, color="#0f172a", family="Inter, sans-serif"),
                hovertext=hovers,
                hoverinfo="text"
            ))

    # Title
    title_text = tree_title
    if path_info:
        title_text += f" (Traversed {len(visited_nodes)} nodes to Leaf #{final_leaf} - {path_info['final_prediction_label']})"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color="#0f172a")),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#334155", size=11),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        dragmode="pan"
    )

    return fig


def explain_gini_calculation(node_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces step-by-step mathematical breakdown of Gini impurity for educational explanation.
    Formula: Gini = 1 - (p_0^2 + p_1^2)
    """
    p0, p1 = node_data["proportions"]
    total = node_data["n_samples"]
    c0, c1 = node_data["class_counts"]

    p0_sq = p0 ** 2
    p1_sq = p1 ** 2
    sum_sq = p0_sq + p1_sq
    gini = 1.0 - sum_sq

    return {
        "n_samples": total,
        "count_no_failure": c0,
        "count_failure": c1,
        "p0": p0,
        "p1": p1,
        "p0_squared": p0_sq,
        "p1_squared": p1_sq,
        "sum_squared": sum_sq,
        "gini": gini
    }
