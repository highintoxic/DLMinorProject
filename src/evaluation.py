"""
evaluation.py — Metrics, confusion matrix, and comparison tables (Task 7)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import tensorflow as tf


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Evaluate a model on the test set and return all metrics.

    Returns:
        dict with keys: accuracy, precision, recall, f1, report_str, y_pred
    """
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  {model.name} — Test Results")
    print(f"{'─'*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"{'─'*50}")
    print(report)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report_str": report,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    figsize: tuple = (10, 8),
) -> None:
    """Plot a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Proportion"},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def build_comparison_table(
    results: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """
    Build a comparison DataFrame across models and settings.

    Args:
        results: nested dict of {setting: {model_name: metrics_dict}}
                 e.g. {"Baseline": {"EfficientNetB0": {"accuracy": 0.85, ...}}}

    Returns:
        pd.DataFrame with multi-index columns
    """
    rows = []
    for setting, models in results.items():
        for model_name, metrics in models.items():
            rows.append({
                "Setting": setting,
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1"],
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["Setting", "Model"])
    return df


def display_comparison(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Pretty-print and optionally save the comparison table."""
    styled = df.style.format("{:.4f}").highlight_max(
        axis=0, props="font-weight: bold; color: green;"
    )
    print("\n" + df.to_string())
    print()

    if save_path:
        df.to_csv(save_path)
        print(f"Saved comparison table to {save_path}")

    return styled
