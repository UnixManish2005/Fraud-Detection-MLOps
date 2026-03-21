"""
feature_engineering.py
-----------------------
Additional feature engineering utilities for the fraud detection pipeline.

These functions are designed to be called AFTER the basic preprocessing
(scaling / SMOTE) is complete and before model training.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ── Feature utilities ──────────────────────────────────────────────────────────

def add_amount_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log1p of the raw (un-scaled) Amount as an extra feature.
    Call this BEFORE scale_amount() if you want the log of the original values.

    Parameters
    ----------
    df : pd.DataFrame  with an 'Amount' column.

    Returns
    -------
    pd.DataFrame
    """
    if "Amount" not in df.columns:
        logger.warning("'Amount' column not found — skipping log transform.")
        return df
    df["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))
    logger.info("Added 'Amount_log' feature.")
    return df


def add_pca_summary(X: np.ndarray, n_components: int = 5) -> np.ndarray:
    """
    Compress the V1-V28 features further into N principal components and
    append them as extra columns.

    Parameters
    ----------
    X            : np.ndarray  — feature matrix (after preprocessing).
    n_components : int         — number of PCA components to append.

    Returns
    -------
    np.ndarray  with n_components extra columns.
    """
    logger.info("Appending %d PCA summary components …", n_components)
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(X)
    return np.hstack([X, pca_features])


def get_feature_names(df_columns: list, use_pca_summary: bool = False, n_pca: int = 5) -> list:
    """
    Return a list of feature names, optionally including PCA summary columns.

    Parameters
    ----------
    df_columns      : list of str  — column names from the DataFrame.
    use_pca_summary : bool         — whether PCA summary was added.
    n_pca           : int          — number of PCA components added.

    Returns
    -------
    list of str
    """
    names = list(df_columns)
    if use_pca_summary:
        names += [f"pca_summary_{i}" for i in range(n_pca)]
    return names


def describe_features(X: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Return a summary DataFrame of feature statistics.

    Parameters
    ----------
    X             : np.ndarray
    feature_names : list of str

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(X, columns=feature_names)
    return df.describe().T
