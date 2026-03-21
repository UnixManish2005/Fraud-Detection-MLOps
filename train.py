"""
train.py
--------
Convenient CLI entry point at the project root.

Usage:
    python train.py                                    # defaults to data/creditcard.csv
    python train.py --data path/to/creditcard.csv
    python train.py --data data/creditcard.csv --no-smote --tune-threshold
    python train.py --evaluate                         # also run full evaluation after training
"""

import argparse
import logging
import os
import sys
import numpy as np

# Make src/ importable
sys.path.insert(0, os.path.dirname(__file__))

from src.model_training import run as train_run
from src.evaluate import full_evaluation

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and (optionally) evaluate the fraud detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/creditcard.csv",
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE; use class_weight='balanced' in models instead.",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune the decision threshold for maximum F2 score.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run full evaluation (plots + SHAP) after training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  🚀  Fraud Detection — Training Pipeline")
    print("=" * 60 + "\n")

    model, model_name, scores, X_test, y_test = train_run(
        data_path=args.data,
        apply_smote=not args.no_smote,
        tune_thresh=args.tune_threshold,
    )

    print("\n── Model Comparison (ROC-AUC) ───────────────────────────────")
    for name, auc in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " ← best" if name == model_name else ""
        print(f"  {name:<25}  AUC = {auc:.4f}{marker}")
    print()

    if args.evaluate:
        print("── Running full evaluation …\n")
        from src.evaluate import full_evaluation
        import joblib
        meta = joblib.load("models/model_meta.joblib")
        threshold = meta.get("threshold", 0.5)
        full_evaluation(model, np.array(X_test), np.array(y_test),
                        model_name=model_name, threshold=threshold)
        print("\nPlots saved to ./plots/")

    print("\n✅  Done!  Model saved to models/best_model.joblib")
    print("   Start API: uvicorn app.main:app --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()
