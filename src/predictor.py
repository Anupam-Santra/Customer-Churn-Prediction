# ============================================================
# src/predictor.py
# Run inference on new / existing customers
# ============================================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from colorama import Fore, Style, init
init(autoreset=True)
import warnings
warnings.filterwarnings("ignore")


class ChurnPredictor:
    """
    Load a trained pipeline (model + encoders + scaler)
    and predict churn probability for new customers.
    """

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.model      = joblib.load(self.models_dir / "best_model.pkl")
        self.encoders   = joblib.load(self.models_dir / "encoders.pkl")
        self.scaler     = joblib.load(self.models_dir / "scaler.pkl")
        print(f"[PREDICTOR] Loaded model: {type(self.model).__name__}")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mirror the feature engineering from preprocessing.py."""
        df = df.copy()
        df["avg_monthly_spend"] = df["total_charges"] / (df["tenure"] + 1)
        df["charge_vs_avg"]     = df["monthly_charges"] - df["monthly_charges"].mean()

        service_cols = ["online_security", "online_backup", "device_protection",
                        "tech_support", "streaming_tv", "streaming_movies"]
        df["num_services"] = df[service_cols].apply(
            lambda row: sum(1 for v in row if str(v).strip() == "Yes"), axis=1
        )
        df["engagement_score"]  = df["tenure"] * (df["num_services"] + 1)
        df["is_high_value"]     = (df["monthly_charges"] > 70).astype(int)
        df["is_month_to_month"] = (df["contract_type"] == "Month-to-month").astype(int)
        df["digital_risk"]      = (
            (df["paperless_billing"] == 1) &
            (df["payment_method"]    == "Electronic check")
        ).astype(int)

        bucket_map = {"0-6m": 0, "6-12m": 1, "1-2yr": 2, "2-4yr": 3, "4yr+": 4}
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[0, 6, 12, 24, 48, 100],
            labels=["0-6m", "6-12m", "1-2yr", "2-4yr", "4yr+"]
        ).astype(str).map(bucket_map).fillna(0).astype(int)

        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        drop_cols = ["customer_id", "churn"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            le = self.encoders.get(col)
            if le:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                X[col] = X[col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0]
                )
                X[col] = le.transform(X[col])
        return X

    def predict(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn for a DataFrame of customers.
        Returns the input DataFrame with added columns:
            churn_probability, churn_prediction, risk_level
        """
        df_feat  = self._engineer_features(customer_data)
        X        = self._encode(df_feat)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

        probs    = self.model.predict_proba(X_scaled)[:, 1]
        preds    = (probs >= 0.50).astype(int)

        result   = customer_data.copy().reset_index(drop=True)
        result["churn_probability"]  = probs.round(4)
        result["churn_prediction"]   = preds
        result["risk_level"]         = pd.cut(
            probs,
            bins=[0, 0.30, 0.60, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )
        return result

    def predict_single(self, customer: dict) -> dict:
        """Predict churn for a single customer given as a dict."""
        df  = pd.DataFrame([customer])
        res = self.predict(df).iloc[0]
        return {
            "customer_id"       : customer.get("customer_id", "N/A"),
            "churn_probability" : float(res["churn_probability"]),
            "churn_prediction"  : int(res["churn_prediction"]),
            "risk_level"        : str(res["risk_level"]),
        }


def print_prediction_report(results: pd.DataFrame):
    """Pretty-print prediction results to console."""
    print("\n" + "="*65)
    print("  CHURN PREDICTION REPORT")
    print("="*65)

    for _, row in results.iterrows():
        cid   = row.get("customer_id", "N/A")
        prob  = row["churn_probability"]
        pred  = row["churn_prediction"]
        risk  = row["risk_level"]

        if risk == "High Risk":
            color = Fore.RED
        elif risk == "Medium Risk":
            color = Fore.YELLOW
        else:
            color = Fore.GREEN

        status = "WILL CHURN" if pred == 1 else "WILL STAY"
        print(f"  Customer: {cid:<12} | "
              f"Probability: {prob:.2%} | "
              f"Prediction: {color}{status:<12}{Style.RESET_ALL} | "
              f"Risk: {color}{risk}{Style.RESET_ALL}")
    print("="*65)

    total      = len(results)
    churners   = results["churn_prediction"].sum()
    high_risk  = (results["risk_level"] == "High Risk").sum()
    print(f"\n  Total Customers : {total}")
    print(f"  Predicted Churn : {churners} ({churners/total*100:.1f}%)")
    print(f"  High Risk       : {high_risk} ({high_risk/total*100:.1f}%)")
    print("="*65 + "\n")


def run_batch_prediction(models_dir: str, data_path: str, output_dir: str):
    """Run predictions on a full CSV file and save results."""
    predictor  = ChurnPredictor(models_dir)
    df         = pd.read_csv(data_path)

    # Remove target column if present (inference mode)
    df_input   = df.drop(columns=["churn"], errors="ignore")

    results    = predictor.predict(df_input)
    print_prediction_report(results.head(20))

    out_path   = Path(output_dir) / "churn_predictions.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"[SAVED] Predictions → {out_path}")
    return results


# ── Sample customers for demo ────────────────────────────────
SAMPLE_CUSTOMERS = [
    {
        "customer_id": "DEMO001", "age": 28, "gender": "Male",
        "senior_citizen": 0, "has_partner": 0, "has_dependents": 0,
        "tenure": 3, "phone_service": 1, "multiple_lines": "No",
        "internet_service": "Fiber optic", "online_security": "No",
        "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "Yes",
        "contract_type": "Month-to-month", "paperless_billing": 1,
        "payment_method": "Electronic check",
        "monthly_charges": 95.40, "total_charges": 286.20
    },
    {
        "customer_id": "DEMO002", "age": 52, "gender": "Female",
        "senior_citizen": 0, "has_partner": 1, "has_dependents": 1,
        "tenure": 48, "phone_service": 1, "multiple_lines": "Yes",
        "internet_service": "DSL", "online_security": "Yes",
        "online_backup": "Yes", "device_protection": "Yes",
        "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No",
        "contract_type": "Two year", "paperless_billing": 0,
        "payment_method": "Bank transfer (automatic)",
        "monthly_charges": 58.50, "total_charges": 2808.0
    },
    {
        "customer_id": "DEMO003", "age": 35, "gender": "Male",
        "senior_citizen": 0, "has_partner": 1, "has_dependents": 0,
        "tenure": 12, "phone_service": 1, "multiple_lines": "No",
        "internet_service": "Fiber optic", "online_security": "No",
        "online_backup": "Yes", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "No",
        "contract_type": "Month-to-month", "paperless_billing": 1,
        "payment_method": "Electronic check",
        "monthly_charges": 78.90, "total_charges": 946.80
    },
    {
        "customer_id": "DEMO004", "age": 64, "gender": "Female",
        "senior_citizen": 1, "has_partner": 0, "has_dependents": 0,
        "tenure": 6, "phone_service": 1, "multiple_lines": "No",
        "internet_service": "Fiber optic", "online_security": "No",
        "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
        "contract_type": "Month-to-month", "paperless_billing": 1,
        "payment_method": "Electronic check",
        "monthly_charges": 69.70, "total_charges": 418.20
    },
    {
        "customer_id": "DEMO005", "age": 41, "gender": "Male",
        "senior_citizen": 0, "has_partner": 1, "has_dependents": 1,
        "tenure": 60, "phone_service": 1, "multiple_lines": "Yes",
        "internet_service": "DSL", "online_security": "Yes",
        "online_backup": "Yes", "device_protection": "Yes",
        "tech_support": "Yes", "streaming_tv": "Yes", "streaming_movies": "Yes",
        "contract_type": "Two year", "paperless_billing": 0,
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 89.20, "total_charges": 5352.0
    },
]


if __name__ == "__main__":
    base      = Path(__file__).parent.parent
    predictor = ChurnPredictor(str(base / "models"))
    df_demo   = pd.DataFrame(SAMPLE_CUSTOMERS)
    results   = predictor.predict(df_demo)
    print_prediction_report(results)
    results.to_csv(base / "outputs" / "demo_predictions.csv", index=False)
