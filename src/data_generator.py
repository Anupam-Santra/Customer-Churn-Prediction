# ============================================================
# src/data_generator.py
# Synthetic Telecom Customer Dataset Generator
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

def generate_churn_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic telecom customer churn dataset.
    Simulates real-world customer behavior patterns.

    Parameters:
        n_samples    : Number of customer records
        random_state : Seed for reproducibility

    Returns:
        pd.DataFrame : Full customer dataset
    """
    np.random.seed(random_state)
    print(f"[INFO] Generating {n_samples} synthetic customer records...")

    # ── Demographics ────────────────────────────────────────
    age              = np.random.randint(18, 75, n_samples)
    gender           = np.random.choice(["Male", "Female"], n_samples)
    senior_citizen   = (age >= 60).astype(int)
    has_partner      = np.random.choice([0, 1], n_samples, p=[0.45, 0.55])
    has_dependents   = np.random.choice([0, 1], n_samples, p=[0.60, 0.40])

    # ── Tenure & Contract ───────────────────────────────────
    contract_type    = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples, p=[0.55, 0.25, 0.20]
    )
    tenure_map       = {"Month-to-month": (1, 30), "One year": (12, 48), "Two year": (24, 72)}
    tenure           = np.array([
        np.random.randint(*tenure_map[c]) for c in contract_type
    ])

    # ── Services ────────────────────────────────────────────
    phone_service        = np.random.choice([0, 1], n_samples, p=[0.10, 0.90])
    multiple_lines       = np.where(phone_service == 1,
                               np.random.choice(["Yes", "No"], n_samples, p=[0.45, 0.55]),
                               "No phone service")
    internet_service     = np.random.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20]
    )
    online_security      = np.where(internet_service != "No",
                               np.random.choice(["Yes", "No"], n_samples, p=[0.40, 0.60]),
                               "No internet service")
    online_backup        = np.where(internet_service != "No",
                               np.random.choice(["Yes", "No"], n_samples, p=[0.42, 0.58]),
                               "No internet service")
    device_protection    = np.where(internet_service != "No",
                               np.random.choice(["Yes", "No"], n_samples, p=[0.38, 0.62]),
                               "No internet service")
    tech_support         = np.where(internet_service != "No",
                               np.random.choice(["Yes", "No"], n_samples, p=[0.35, 0.65]),
                               "No internet service")
    streaming_tv         = np.where(internet_service != "No",
                               np.random.choice(["Yes", "No"], n_samples, p=[0.50, 0.50]),
                               "No internet service")
    streaming_movies     = np.where(internet_service != "No",
                               np.random.choice(["Yes", "No"], n_samples, p=[0.50, 0.50]),
                               "No internet service")

    # ── Billing ─────────────────────────────────────────────
    paperless_billing    = np.random.choice([0, 1], n_samples, p=[0.40, 0.60])
    payment_method       = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_samples, p=[0.35, 0.20, 0.23, 0.22]
    )

    # Monthly charges driven by services
    base_charge          = np.random.uniform(20, 40, n_samples)
    internet_surcharge   = np.where(internet_service == "Fiber optic", 30,
                           np.where(internet_service == "DSL", 15, 0))
    service_surcharge    = (
        (multiple_lines   == "Yes").astype(int) * 10 +
        (online_security  == "Yes").astype(int) * 5  +
        (online_backup    == "Yes").astype(int) * 5  +
        (device_protection== "Yes").astype(int) * 5  +
        (tech_support     == "Yes").astype(int) * 5  +
        (streaming_tv     == "Yes").astype(int) * 8  +
        (streaming_movies == "Yes").astype(int) * 8
    )
    noise                = np.random.normal(0, 5, n_samples)
    monthly_charges      = np.clip(base_charge + internet_surcharge + service_surcharge + noise, 18, 120)
    total_charges        = monthly_charges * tenure + np.random.normal(0, 50, n_samples)
    total_charges        = np.clip(total_charges, 0, None)

    # ── Churn Logic (realistic probabilities) ───────────────
    churn_score = np.zeros(n_samples)

    # High monthly charge → more likely to churn
    churn_score += (monthly_charges > 70) * 0.30
    churn_score += (monthly_charges > 90) * 0.20

    # Short tenure → more likely to churn
    churn_score += (tenure < 6)  * 0.35
    churn_score += (tenure < 12) * 0.15

    # Month-to-month contract → very likely to churn
    churn_score += (contract_type == "Month-to-month") * 0.35

    # No tech support / security → more likely to churn
    churn_score += (online_security == "No") * 0.10
    churn_score += (tech_support    == "No") * 0.10

    # Fiber optic → slightly higher churn (price sensitivity)
    churn_score += (internet_service == "Fiber optic") * 0.15

    # Electronic check → higher churn (less automated)
    churn_score += (payment_method == "Electronic check") * 0.10

    # Senior citizen → slightly more likely to churn
    churn_score += senior_citizen * 0.08

    # Dependent / partner → less likely to churn (stickiness)
    churn_score -= has_partner     * 0.10
    churn_score -= has_dependents  * 0.08

    # Long-term contract → much less likely to churn
    churn_score -= (contract_type == "One year")  * 0.20
    churn_score -= (contract_type == "Two year")  * 0.35

    # Clip to valid probability range
    churn_prob  = np.clip(churn_score, 0.03, 0.95)
    churn       = (np.random.uniform(0, 1, n_samples) < churn_prob).astype(int)

    # ── Assemble DataFrame ──────────────────────────────────
    df = pd.DataFrame({
        "customer_id"         : [f"CUST{str(i).zfill(5)}" for i in range(1, n_samples + 1)],
        "age"                 : age,
        "gender"              : gender,
        "senior_citizen"      : senior_citizen,
        "has_partner"         : has_partner,
        "has_dependents"      : has_dependents,
        "tenure"              : tenure,
        "phone_service"       : phone_service,
        "multiple_lines"      : multiple_lines,
        "internet_service"    : internet_service,
        "online_security"     : online_security,
        "online_backup"       : online_backup,
        "device_protection"   : device_protection,
        "tech_support"        : tech_support,
        "streaming_tv"        : streaming_tv,
        "streaming_movies"    : streaming_movies,
        "contract_type"       : contract_type,
        "paperless_billing"   : paperless_billing,
        "payment_method"      : payment_method,
        "monthly_charges"     : monthly_charges.round(2),
        "total_charges"       : total_charges.round(2),
        "churn"               : churn
    })

    churn_rate = churn.mean() * 100
    print(f"[INFO] Dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[INFO] Churn rate: {churn_rate:.2f}% ({churn.sum()} churned / {n_samples - churn.sum()} retained)")
    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "data" / "customer_churn_raw.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_churn_dataset(n_samples=10000)
    df.to_csv(output_path, index=False)
    print(f"[SAVED] → {output_path}")
    print(df.head())
