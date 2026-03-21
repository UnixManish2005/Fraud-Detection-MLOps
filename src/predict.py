"""
predict.py
----------
Inference utilities used both by the FastAPI app and standalone CLI predictions.

Usage (CLI):
    python src/predict.py --input '{"V1": -1.36, "V2": -0.07, ..., "Amount": 149.62}'
"""

import argparse
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR      = os.path.join(os.path.dirname(__file__), "..", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
META_PATH       = os.path.join(MODELS_DIR, "model_meta.joblib")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.joblib")

# ── Expected raw feature columns (V1-V28 + Amount + optionally Time) ──────────
V_FEATURES   = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = V_FEATURES + ["Amount"]  # Time is engineered away


# ── Model loader ───────────────────────────────────────────────────────────────

class FraudPredictor:
    """
    Encapsulates model + scaler + metadata for serving predictions.
    Loaded once at startup; thread-safe for inference.
    """

    def __init__(self):
        self._model   = None
        self._scaler  = None
        self._meta    = {}
        self._loaded  = False

    def load(self):
        """Load model, scaler, and metadata from disk."""
        if self._loaded:
            return

        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {BEST_MODEL_PATH}. "
                "Run model_training.py first."
            )

        self._model  = joblib.load(BEST_MODEL_PATH)
        self._scaler = joblib.load(SCALER_PATH)
        self._meta   = joblib.load(META_PATH)
        self._loaded = True
        logger.info(
            "FraudPredictor loaded: %s  |  threshold=%.2f",
            self._meta.get("model_name", "unknown"),
            self._meta.get("threshold", 0.5),
        )

    @property
    def threshold(self) -> float:
        return self._meta.get("threshold", 0.5)

    @property
    def model_name(self) -> str:
        return self._meta.get("model_name", "unknown")

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self, raw: dict) -> np.ndarray:
        """
        Transform a raw feature dict into the numpy array the model expects.
        Applies the same transformations as the training pipeline:
          1. Engineer cyclic hour features from 'Time' (if present)
          2. Scale 'Amount' with the saved StandardScaler

        Parameters
        ----------
        raw : dict  — keys include V1-V28, Amount, and optionally Time.

        Returns
        -------
        np.ndarray  shape (1, n_features)
        """
        # Build a single-row DataFrame with all raw columns
        df = pd.DataFrame([raw])

        # Engineer Time → hour_sin, hour_cos  (if Time is present)
        if "Time" in df.columns:
            seconds_in_day = 24 * 3600
            angle = (df["Time"] % seconds_in_day) / seconds_in_day * 2 * np.pi
            df["hour_sin"] = np.sin(angle)
            df["hour_cos"] = np.cos(angle)
            df = df.drop(columns=["Time"])
        else:
            # Provide zero placeholders so feature count stays consistent
            df["hour_sin"] = 0.0
            df["hour_cos"] = 0.0

        # Scale Amount
        df["Amount"] = self._scaler.transform(df[["Amount"]])

        return df.values

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, raw: dict) -> dict:
        """
        Run inference on a single transaction dict.

        Parameters
        ----------
        raw : dict  — raw feature values (V1-V28, Amount, optional Time)

        Returns
        -------
        dict with keys:
            fraud_probability : float
            is_fraud          : bool
            model_name        : str
            threshold         : float
        """
        if not self._loaded:
            self.load()

        X = self._preprocess(raw)
        proba = float(self._model.predict_proba(X)[0, 1])
        is_fraud = proba >= self.threshold

        return {
            "fraud_probability": round(proba, 6),
            "is_fraud": bool(is_fraud),
            "model_name": self.model_name,
            "threshold": self.threshold,
        }

    def predict_batch(self, records: list) -> list:
        """
        Run inference on a list of raw transaction dicts.

        Parameters
        ----------
        records : list of dict

        Returns
        -------
        list of result dicts (same structure as predict())
        """
        if not self._loaded:
            self.load()

        rows = [self._preprocess(r) for r in records]
        X = np.vstack(rows)
        probas = self._model.predict_proba(X)[:, 1]

        return [
            {
                "fraud_probability": round(float(p), 6),
                "is_fraud": bool(p >= self.threshold),
                "model_name": self.model_name,
                "threshold": self.threshold,
            }
            for p in probas
        ]


# ── Singleton instance (shared by FastAPI) ─────────────────────────────────────
predictor = FraudPredictor()


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single fraud prediction from the CLI.")
    parser.add_argument(
        "--input",
        required=True,
        help='JSON string of feature values, e.g. \'{"V1": -1.36, ..., "Amount": 149.62}\'',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw = json.loads(args.input)
    predictor.load()
    result = predictor.predict(raw)
    print(json.dumps(result, indent=2))
