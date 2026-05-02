# ============================================================
# main.py
# Master Orchestration Script
# Run this first to generate all outputs.
# Usage: python main.py
# ============================================================

import sys
import time
from pathlib import Path
from colorama import Fore, Style, init
init(autoreset=True)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "src"))

# ── Import pipeline modules ──────────────────────────────────
from src.data_generator    import generate_churn_dataset
from src.preprocessing     import run_full_pipeline
from src.model_training    import run_training_pipeline
from src.visualization     import run_all_eda, run_all_evaluation
from src.explainability    import run_explainability
from src.predictor         import ChurnPredictor, SAMPLE_CUSTOMERS, print_prediction_report
from src.business_insights import (calculate_business_metrics, print_business_report,
                                    plot_business_dashboard)

import pandas as pd
import joblib

# ── Banner ───────────────────────────────────────────────────
BANNER = f"""
{Fore.CYAN}
╔══════════════════════════════════════════════════════════════╗
║       CUSTOMER CHURN PREDICTION MODEL  — OPTION C           ║
║       Advanced ML Pipeline | Placement Ready                 ║
╚══════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""

def step(n, title):
    print(f"\n{Fore.YELLOW}{'─'*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'─'*60}{Style.RESET_ALL}")


def main():
    t_start = time.time()
    print(BANNER)

    # ── Directories ──────────────────────────────────────────
    DATA_DIR    = BASE / "data"
    MODELS_DIR  = BASE / "models"
    IMAGES_DIR  = BASE / "images"
    OUTPUTS_DIR = BASE / "outputs"
    for d in [DATA_DIR, MODELS_DIR, IMAGES_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════
    # STEP 1 — Generate Synthetic Dataset
    # ════════════════════════════════════════════════════════
    step(1, "Generating Synthetic Dataset")
    df = generate_churn_dataset(n_samples=10000, random_state=42)
    raw_path = DATA_DIR / "customer_churn_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"{Fore.GREEN}[✓] Dataset saved → {raw_path}{Style.RESET_ALL}")

    # ════════════════════════════════════════════════════════
    # STEP 2 — EDA Visualizations (on raw data)
    # ════════════════════════════════════════════════════════
    step(2, "Generating EDA Visualizations")
    run_all_eda(df, str(IMAGES_DIR))
    print(f"{Fore.GREEN}[✓] EDA plots saved → {IMAGES_DIR}{Style.RESET_ALL}")

    # ════════════════════════════════════════════════════════
    # STEP 3 — Preprocessing Pipeline
    # ════════════════════════════════════════════════════════
    step(3, "Running Preprocessing Pipeline")
    X_train, X_test, y_train, y_test, encoders, scaler = run_full_pipeline(
        raw_path   = str(raw_path),
        output_dir = str(DATA_DIR)
    )
    print(f"{Fore.GREEN}[✓] Preprocessing complete.{Style.RESET_ALL}")

    # ════════════════════════════════════════════════════════
    # STEP 4 — Model Training & Evaluation
    # ════════════════════════════════════════════════════════
    step(4, "Training All Models")
    trained_models, results, best_name, X_test_df, y_test_s = run_training_pipeline(
        data_dir   = str(DATA_DIR),
        models_dir = str(MODELS_DIR)
    )
    print(f"{Fore.GREEN}[✓] Best model: {best_name}{Style.RESET_ALL}")

    # ════════════════════════════════════════════════════════
    # STEP 5 — Model Evaluation Plots
    # ════════════════════════════════════════════════════════
    step(5, "Generating Model Evaluation Plots")
    rf_model  = trained_models.get("Random Forest")
    feat_names = X_test_df.columns.tolist()
    run_all_evaluation(results, y_test_s, rf_model, feat_names, str(IMAGES_DIR))
    print(f"{Fore.GREEN}[✓] Evaluation plots saved → {IMAGES_DIR}{Style.RESET_ALL}")

    # ════════════════════════════════════════════════════════
    # STEP 6 — SHAP Explainability
    # ════════════════════════════════════════════════════════
    step(6, "Running SHAP Explainability")
    best_model  = trained_models[best_name]
    shap_values, top_features = run_explainability(
        model      = best_model,
        model_name = best_name,
        X_test     = X_test_df,
        output_dir = str(IMAGES_DIR),
        sample_size = 500
    )
    top_features.to_csv(OUTPUTS_DIR / "shap_top_features.csv", index=False)
    print(f"{Fore.GREEN}[✓] SHAP plots saved → {IMAGES_DIR}{Style.RESET_ALL}")

    # ════════════════════════════════════════════════════════
    # STEP 7 — Batch Predictions (Test Set)
    # ════════════════════════════════════════════════════════
    step(7, "Running Batch Predictions")
    predictor   = ChurnPredictor(str(MODELS_DIR))
    raw_test    = pd.read_csv(raw_path)
    test_input  = raw_test.drop(columns=["churn"], errors="ignore")
    predictions = predictor.predict(test_input)
    pred_path   = OUTPUTS_DIR / "churn_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    print_prediction_report(predictions.head(10))
    print(f"{Fore.GREEN}[✓] Predictions saved → {pred_path}{Style.RESET_ALL}")

    # ════════════════════════════════════════════════════════
    # STEP 8 — Demo: Single Customer Predictions
    # ════════════════════════════════════════════════════════
    step(8, "Demo — 5 Sample Customer Predictions")
    demo_df     = pd.DataFrame(SAMPLE_CUSTOMERS)
    demo_results = predictor.predict(demo_df)
    print_prediction_report(demo_results)
    demo_results.to_csv(OUTPUTS_DIR / "demo_predictions.csv", index=False)

    # ════════════════════════════════════════════════════════
    # STEP 9 — Business Intelligence Report
    # ════════════════════════════════════════════════════════
    step(9, "Generating Business Intelligence Report")
    biz_metrics = calculate_business_metrics(predictions)
    print_business_report(biz_metrics)
    plot_business_dashboard(biz_metrics, top_features, IMAGES_DIR)

    # Save business summary
    pd.DataFrame([biz_metrics]).to_csv(OUTPUTS_DIR / "business_summary.csv", index=False)
    print(f"{Fore.GREEN}[✓] Business dashboard saved → {IMAGES_DIR}{Style.RESET_ALL}")

    # ════════════════════════════════════════════════════════
    # DONE
    # ════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{Fore.CYAN}{'═'*60}")
    print(f"  ✅  PIPELINE COMPLETE  —  {elapsed:.1f}s")
    print(f"{'═'*60}{Style.RESET_ALL}")
    print(f"\n  📁 Data       → {DATA_DIR}")
    print(f"  🤖 Models     → {MODELS_DIR}")
    print(f"  🖼  Images     → {IMAGES_DIR}")
    print(f"  📊 Outputs    → {OUTPUTS_DIR}")
    print(f"\n  🚀 To launch the dashboard run:")
    print(f"     {Fore.YELLOW}python src/dashboard.py{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
