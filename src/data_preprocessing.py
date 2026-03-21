"""
data_preprocessing.py
---------------------
Handles loading, cleaning, scaling, and splitting the credit card fraud dataset.
"""

import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE     = 0.20
SCALER_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")


# ── Public API ─────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Loading dataset from: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Class distribution:\n%s", df["Class"].value_counts())
    return df


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log and report missing values.  Drops any fully-empty rows.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  (possibly with empty rows removed)
    """
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values detected:\n%s", missing[missing > 0])
        before = len(df)
        df = df.dropna()
        logger.info("Dropped %d rows with missing values.", before - len(df))
    else:
        logger.info("No missing values found.")
    return df


def engineer_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw 'Time' column (seconds since first transaction) into
    two cyclic features: hour_sin and hour_cos.
    The original 'Time' column is then dropped.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Engineering time features …")
    seconds_in_day = 24 * 3600
    hour_angle = (df["Time"] % seconds_in_day) / seconds_in_day * 2 * np.pi
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df = df.drop(columns=["Time"])
    logger.info("Replaced 'Time' with cyclic hour_sin / hour_cos features.")
    return df


def scale_amount(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Scale the 'Amount' column with StandardScaler.

    Parameters
    ----------
    df  : pd.DataFrame
    fit : bool
        If True, fit a new scaler and save it to disk.
        If False, load the previously saved scaler and only transform.

    Returns
    -------
    pd.DataFrame
    """
    scaler_path = SCALER_PATH
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    if fit:
        logger.info("Fitting and saving StandardScaler for 'Amount' …")
        scaler = StandardScaler()
        df["Amount"] = scaler.fit_transform(df[["Amount"]])
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler saved to %s", scaler_path)
    else:
        logger.info("Loading saved StandardScaler …")
        scaler = joblib.load(scaler_path)
        df["Amount"] = scaler.transform(df[["Amount"]])

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = TEST_SIZE,
    apply_smote: bool = True,
) -> tuple:
    """
    Split into train / test sets and optionally apply SMOTE on the training set.

    Parameters
    ----------
    df          : pd.DataFrame
    target_col  : str
    test_size   : float
    apply_smote : bool
        If True, apply SMOTE oversampling to address class imbalance.
        If False, rely on class_weight='balanced' inside the models instead.

    Returns
    -------
    X_train, X_test, y_train, y_test  (all as numpy arrays or DataFrames)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info("Splitting data — test_size=%.2f, random_state=%d", test_size, RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    if apply_smote:
        logger.info("Applying SMOTE to training set …")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(
            "After SMOTE — Class distribution in train set: %s",
            dict(zip(*np.unique(y_train, return_counts=True))),
        )

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(
    filepath: str,
    apply_smote: bool = True,
    fit_scaler: bool = True,
) -> tuple:
    """
    Full preprocessing pipeline:
        load → check_missing → engineer_time → scale_amount → split

    Parameters
    ----------
    filepath    : str   Path to the raw CSV.
    apply_smote : bool  Whether to apply SMOTE.
    fit_scaler  : bool  Whether to fit (True) or load (False) the scaler.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_data(filepath)
    df = check_missing(df)
    df = engineer_time(df)
    df = scale_amount(df, fit=fit_scaler)
    return split_data(df, apply_smote=apply_smote)
