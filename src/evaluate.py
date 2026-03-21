"""
evaluate.py
-----------
Evaluation utilities: metrics, confusion matrix, ROC curve,
Precision-Recall curve, and SHAP explainability.

Usage (CLI):
    python src/evaluate.py --data data/creditcard.csv
"""

import argparse
import logging
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preprocessing import preprocess_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Output directories ─────────────────────────────────────────────────────────
PLOTS_DIR = "plots"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
META_PATH       = os.path.join(MODELS_DIR, "model_meta.joblib")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def print_metrics(y_true, y_pred, y_proba, model_name: str = "") -> None:
    """
    Print classification report and ROC-AUC to the console.
    """
    header = f"=== {model_name} Evaluation ===" if model_name else "=== Evaluation ==="
    logger.info(header)
    logger.info(
        "\n%s",
        classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]),
    )
    auc = roc_auc_score(y_true, y_proba)
    ap  = average_precision_score(y_true, y_proba)
    logger.info("ROC-AUC                : %.4f", auc)
    logger.info("Average Precision (AP) : %.4f", ap)


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, model_name: str = "model") -> str:
    """
    Save and return the path to the confusion matrix heatmap.
    """
    _ensure_plots_dir()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved → %s", path)
    return path


def plot_roc_curve(y_true, y_proba, model_name: str = "model") -> str:
    """
    Save and return the path to the ROC curve plot.
    """
    _ensure_plots_dir()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = os.path.join(PLOTS_DIR, f"roc_curve_{model_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("ROC curve saved → %s", path)
    return path


def plot_precision_recall_curve(y_true, y_proba, model_name: str = "model") -> str:
    """
    Save and return the path to the Precision-Recall curve plot.
    """
    _ensure_plots_dir()
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    path = os.path.join(PLOTS_DIR, f"pr_curve_{model_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Precision-Recall curve saved → %s", path)
    return path


def plot_shap_summary(model, X_sample: np.ndarray, feature_names: list, model_name: str = "model") -> str:
    """
    Compute SHAP values for a sample of X and save a summary bar plot.

    Parameters
    ----------
    model        : fitted estimator  (must expose predict_proba)
    X_sample     : np.ndarray        — subset of test data (≤ 500 rows for speed)
    feature_names: list of str
    model_name   : str

    Returns
    -------
    str  — path to the saved plot
    """
    _ensure_plots_dir()
    logger.info("Computing SHAP values for %d samples …", len(X_sample))

    # Use TreeExplainer for tree-based models, KernelExplainer as fallback
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For binary classifiers, shap_values is a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        logger.warning("TreeExplainer failed — falling back to KernelExplainer (slower).")
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)[:, :, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    path = os.path.join(PLOTS_DIR, f"shap_summary_{model_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved → %s", path)
    return path


def plot_shap_beeswarm(model, X_sample: np.ndarray, feature_names: list, model_name: str = "model") -> str:
    """
    SHAP beeswarm plot showing individual sample impacts.
    """
    _ensure_plots_dir()
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)[:, :, 1]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False,
    )
    path = os.path.join(PLOTS_DIR, f"shap_beeswarm_{model_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP beeswarm plot saved → %s", path)
    return path


# ── Full evaluation run ────────────────────────────────────────────────────────

def full_evaluation(model, X_test, y_test, model_name: str = "best_model", threshold: float = 0.5):
    """
    Run all evaluation steps and generate all plots.

    Parameters
    ----------
    model       : fitted estimator
    X_test      : test features (numpy array)
    y_test      : true labels
    model_name  : str
    threshold   : float  — decision threshold

    Returns
    -------
    dict with all metric values
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    # Console metrics
    print_metrics(y_test, y_pred, y_proba, model_name=model_name)

    # Plots
    plot_confusion_matrix(y_test, y_pred, model_name=model_name)
    plot_roc_curve(y_test, y_proba, model_name=model_name)
    plot_precision_recall_curve(y_test, y_proba, model_name=model_name)

    # SHAP — limit to 500 samples for speed
    sample_size = min(500, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), sample_size, replace=False)
    X_shap = X_test[idx] if isinstance(X_test, np.ndarray) else X_test.iloc[idx].values
    feature_names = [f"f{i}" for i in range(X_shap.shape[1])]

    plot_shap_summary(model, X_shap, feature_names, model_name=model_name)
    plot_shap_beeswarm(model, X_shap, feature_names, model_name=model_name)

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the best saved fraud model.")
    parser.add_argument("--data", default="data/creditcard.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Preprocess
    _, X_test, _, y_test = preprocess_pipeline(args.data, apply_smote=False, fit_scaler=False)

    # Load saved model
    model = joblib.load(BEST_MODEL_PATH)
    meta  = joblib.load(META_PATH)
    model_name = meta.get("model_name", "best_model")
    threshold  = meta.get("threshold", 0.5)

    logger.info("Loaded model: %s  |  threshold: %.2f", model_name, threshold)
    full_evaluation(model, np.array(X_test), np.array(y_test), model_name=model_name, threshold=threshold)
