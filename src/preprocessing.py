# ============================================================
# src/preprocessing.py
# Data Cleaning, Encoding & Feature Engineering Pipeline
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")


# ── Label maps for consistent encoding ──────────────────────
BINARY_YES_NO_COLS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "multiple_lines"
]

LABEL_ENCODE_COLS = [
    "gender", "internet_service", "contract_type",
    "payment_method", "multiple_lines",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies"
]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV and do basic type fixes."""
    df = pd.read_csv(filepath)
    print(f"[LOAD] Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing / null values."""
    before = df.isnull().sum().sum()
    # Total charges might be blank for new customers
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"].fillna(df["monthly_charges"], inplace=True)
    after = df.isnull().sum().sum()
    print(f"[MISSING] Nulls before: {before} → after: {after}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features that improve model signal.
    These mimic what a real DS team would derive.
    """
    df = df.copy()

    # Average monthly spend normalised by tenure
    df["avg_monthly_spend"]       = df["total_charges"] / (df["tenure"] + 1)

    # Charge deviation (how much they pay vs average)
    df["charge_vs_avg"]           = df["monthly_charges"] - df["monthly_charges"].mean()

    # Number of add-on services subscribed
    service_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if str(v).strip() == "Yes"), axis=1
    )

    # Engagement score: tenure × services
    df["engagement_score"]        = df["tenure"] * (df["num_services"] + 1)

    # High-value customer flag
    df["is_high_value"]           = (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)).astype(int)

    # Contract risk flag (month-to-month = high risk)
    df["is_month_to_month"]       = (df["contract_type"] == "Month-to-month").astype(int)

    # Paperless + electronic check (digital footprint risk)
    df["digital_risk"]            = (
        (df["paperless_billing"] == 1) &
        (df["payment_method"]    == "Electronic check")
    ).astype(int)

    # Tenure bucket
    df["tenure_bucket"]           = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 100],
        labels=["0-6m", "6-12m", "1-2yr", "2-4yr", "4yr+"]
    )

    print(f"[FEATURES] New features created: avg_monthly_spend, charge_vs_avg, num_services, "
          f"engagement_score, is_high_value, is_month_to_month, digital_risk, tenure_bucket")
    return df


def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """
    Label-encode categorical columns.
    If fit=True (training), fits new encoders.
    If fit=False (inference), reuses saved encoders.
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

    # Drop ID and target
    drop_cols = ["customer_id", "churn"] if "churn" in df.columns else ["customer_id"]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Drop tenure_bucket (ordinal string) - convert separately
    if "tenure_bucket" in X.columns:
        bucket_map = {"0-6m": 0, "6-12m": 1, "1-2yr": 2, "2-4yr": 3, "4yr+": 4}
        X["tenure_bucket"] = X["tenure_bucket"].astype(str).map(bucket_map).fillna(0).astype(int)

    # Label encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                X[col] = le.transform(X[col].astype(str))

    y = df["churn"] if "churn" in df.columns else None

    print(f"[ENCODE] Encoded {len(cat_cols)} categorical columns")
    return X, y, encoders


def scale_features(X_train, X_test, scaler_type: str = "standard") -> tuple:
    """Scale numerical features."""
    Scaler = StandardScaler if scaler_type == "standard" else MinMaxScaler
    scaler = Scaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    print(f"[SCALE] Applied {scaler_type} scaling")
    return X_train_scaled, X_test_scaled, scaler


def run_full_pipeline(raw_path: str, output_dir: str, test_size: float = 0.20):
    """
    Master pipeline: load → clean → feature engineer → encode → split → scale → save.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  PREPROCESSING PIPELINE")
    print("="*60)

    df = load_raw_data(raw_path)
    df = handle_missing_values(df)
    df = engineer_features(df)

    # Save engineered dataset
    engineered_path = output_dir / "customer_churn_engineered.csv"
    df.to_csv(engineered_path, index=False)
    print(f"[SAVED] Engineered data → {engineered_path}")

    X, y, encoders = encode_features(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[SPLIT] Train: {X_train.shape} | Test: {X_test.shape}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save splits
    X_train_scaled.to_csv(output_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv( output_dir / "X_test.csv",  index=False)
    y_train.to_csv(        output_dir / "y_train.csv", index=False)
    y_test.to_csv(         output_dir / "y_test.csv",  index=False)

    # Save artifacts
    models_dir = Path(output_dir).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, models_dir / "encoders.pkl")
    joblib.dump(scaler,   models_dir / "scaler.pkl")
    print(f"[SAVED] Encoders → {models_dir / 'encoders.pkl'}")
    print(f"[SAVED] Scaler   → {models_dir / 'scaler.pkl'}")

    print("\n[DONE] Preprocessing complete!")
    print(f"       Features: {X_train.shape[1]} | Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("="*60 + "\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, encoders, scaler


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    run_full_pipeline(
        raw_path   = str(base / "data" / "customer_churn_raw.csv"),
        output_dir = str(base / "data")
    )
