"""
model_training.py
-----------------
Train Logistic Regression, Random Forest, and XGBoost models.
Compare performance and save the best model to disk.

Usage (CLI):
    python src/model_training.py --data data/creditcard.csv
    python src/model_training.py --data data/creditcard.csv --no-smote
"""

import argparse
import logging
import os
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ── Path fix so src/ imports work when run from project root ───────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preprocessing import preprocess_pipeline

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "training.log")),
    ],
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
META_PATH       = os.path.join(MODELS_DIR, "model_meta.joblib")


# ── Model definitions ──────────────────────────────────────────────────────────

def get_model_definitions() -> dict:
    """
    Return a dict of {name: unfitted_estimator} for the three candidate models.
    class_weight / scale_pos_weight keep things fair on imbalanced data in case
    SMOTE is not applied.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=100,   # rough inverse of fraud ratio
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# ── Training helpers ───────────────────────────────────────────────────────────

def train_models(X_train, y_train) -> dict:
    """
    Fit all candidate models on the training data.

    Returns
    -------
    dict  {model_name: fitted_estimator}
    """
    models = get_model_definitions()
    fitted = {}

    for name, model in models.items():
        logger.info("Training %s …", name)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        logger.info("%s trained in %.1f s", name, elapsed)
        fitted[name] = model

    return fitted


def select_best_model(fitted_models: dict, X_test, y_test) -> tuple:
    """
    Evaluate each model by ROC-AUC on the test set and return the best one.

    Returns
    -------
    (best_name: str, best_model: estimator, scores: dict)
    """
    scores = {}
    logger.info("Evaluating models on test set …")

    for name, model in fitted_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        scores[name] = auc
        logger.info("  %-25s  ROC-AUC = %.4f", name, auc)

    best_name = max(scores, key=scores.get)
    logger.info("Best model: %s  (AUC = %.4f)", best_name, scores[best_name])
    return best_name, fitted_models[best_name], scores


def save_model(model, model_name: str, feature_names: list, threshold: float = 0.5) -> None:
    """
    Persist the best model and its metadata to disk.

    Parameters
    ----------
    model        : fitted estimator
    model_name   : str   human-readable name
    feature_names: list  column names used during training
    threshold    : float decision threshold for fraud classification
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)

    meta = {
        "model_name": model_name,
        "feature_names": feature_names,
        "threshold": threshold,
    }
    joblib.dump(meta, META_PATH)

    logger.info("Model saved  → %s", BEST_MODEL_PATH)
    logger.info("Meta saved   → %s", META_PATH)


def tune_threshold(model, X_val, y_val, beta: float = 2.0) -> float:
    """
    Find the probability threshold that maximises F-beta score (default β=2
    so recall is weighted more than precision — critical for fraud detection).

    Parameters
    ----------
    model  : fitted estimator
    X_val  : validation features
    y_val  : true labels
    beta   : float — weight of recall in F-beta

    Returns
    -------
    float — optimal threshold
    """
    from sklearn.metrics import fbeta_score

    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thresh, best_score = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = fbeta_score(y_val, y_pred, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_thresh = t

    logger.info(
        "Optimal threshold (F%.1f): %.2f  (F%.1f = %.4f)",
        beta, best_thresh, beta, best_score,
    )
    return float(best_thresh)


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--data", default="data/creditcard.csv",
        help="Path to the raw creditcard.csv file.",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE; rely on class_weight='balanced' instead.",
    )
    parser.add_argument(
        "--tune-threshold", action="store_true",
        help="Run threshold tuning after model selection.",
    )
    return parser.parse_args()


def run(data_path: str, apply_smote: bool = True, tune_thresh: bool = False):
    """
    Full training run.  Can be called programmatically or from __main__.
    """
    logger.info("=" * 60)
    logger.info("Fraud Detection — Model Training")
    logger.info("=" * 60)

    # 1. Preprocess
    X_train, X_test, y_train, y_test = preprocess_pipeline(
        data_path, apply_smote=apply_smote
    )
    feature_names = [f"f{i}" for i in range(X_train.shape[1])]  # placeholder names

    # 2. Train
    fitted_models = train_models(X_train, y_train)

    # 3. Select best
    best_name, best_model, scores = select_best_model(fitted_models, X_test, y_test)

    # 4. Threshold tuning
    threshold = 0.5
    if tune_thresh:
        threshold = tune_threshold(best_model, X_test, y_test)

    # 5. Save
    save_model(best_model, best_name, feature_names, threshold)

    logger.info("Training complete.")
    return best_model, best_name, scores, X_test, y_test


if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=args.data,
        apply_smote=not args.no_smote,
        tune_thresh=args.tune_threshold,
    )
