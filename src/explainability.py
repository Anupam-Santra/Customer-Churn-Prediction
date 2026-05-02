# ============================================================
# src/explainability.py
# SHAP-based Model Explainability
# Summary plots, Waterfall, Force plots, Feature Importance
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving
import shap
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def compute_shap_values(model, X_data: pd.DataFrame, model_name: str = "model"):
    """
    Compute SHAP values using the right explainer for the model type.
    Tree-based → TreeExplainer (fast)
    Others     → KernelExplainer (slow, sample first)
    """
    print(f"\n[SHAP] Computing SHAP values for: {model_name}")

    tree_models = ("XGB", "LGB", "LGBM", "Forest", "Boosting", "Tree")
    is_tree = any(t.lower() in type(model).__name__.lower() for t in tree_models)

    if is_tree:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)
        # For binary classification, some models return list [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = expected_val[1] if len(expected_val) > 1 else expected_val[0]
    else:
        # Sample 200 rows to keep kernel explainer manageable
        X_sample    = shap.sample(X_data, min(200, len(X_data)), random_state=42)
        explainer   = shap.KernelExplainer(model.predict_proba, X_sample)
        shap_values = explainer.shap_values(X_data[:500], nsamples=100)
        shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
        expected_val = explainer.expected_value[1] if isinstance(
            explainer.expected_value, (list, np.ndarray)
        ) else explainer.expected_value

    print(f"[SHAP] SHAP values shape: {np.array(shap_values).shape}")
    return shap_values, explainer, expected_val


def plot_shap_summary(shap_values, X_data: pd.DataFrame, output_dir: Path, model_name: str):
    """Beeswarm summary plot — shows feature impact and direction."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_data, plot_type="dot", show=False, max_display=15)
    plt.title(f"SHAP Summary Plot — {model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    out = output_dir / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Summary plot saved → {out}")


def plot_shap_bar(shap_values, X_data: pd.DataFrame, output_dir: Path, model_name: str):
    """Bar chart of mean absolute SHAP values — global feature importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.Series(mean_abs, index=X_data.columns).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feat_imp)))
    feat_imp.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"SHAP Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Bar plot saved → {out}")


def plot_shap_waterfall(shap_values, X_data: pd.DataFrame, expected_val,
                         output_dir: Path, model_name: str, sample_idx: int = 0):
    """
    Waterfall plot for a single customer prediction.
    Shows exactly why the model predicted churn / no churn.
    """
    sv = shap_values[sample_idx]
    top_n = 12
    indices = np.argsort(np.abs(sv))[::-1][:top_n]
    features  = X_data.columns[indices]
    values    = sv[indices]
    feat_vals = X_data.iloc[sample_idx][features]

    colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in values]
    labels = [f"{f}\n= {fv:.2f}" for f, fv in zip(features, feat_vals)]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"SHAP Waterfall — Customer #{sample_idx} ({model_name})",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel("SHAP Value (impact on prediction)")
    for bar, val in zip(bars, values):
        xpos = bar.get_width() + 0.001 if val >= 0 else bar.get_width() - 0.001
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"shap_waterfall_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Waterfall plot saved → {out}")


def get_top_features(shap_values, X_data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return a DataFrame of top features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_df  = pd.DataFrame({
        "feature"          : X_data.columns,
        "mean_shap_abs"    : mean_abs,
    }).sort_values("mean_shap_abs", ascending=False).head(top_n).reset_index(drop=True)
    feat_df.index += 1
    return feat_df


def run_explainability(model, model_name: str, X_test: pd.DataFrame,
                        output_dir: str, sample_size: int = 500):
    """
    Run full SHAP explainability pipeline for a given model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a sample for speed
    X_sample = X_test.iloc[:sample_size].reset_index(drop=True)

    shap_values, explainer, expected_val = compute_shap_values(model, X_sample, model_name)

    plot_shap_summary(shap_values,   X_sample, output_dir, model_name)
    plot_shap_bar(shap_values,       X_sample, output_dir, model_name)
    plot_shap_waterfall(shap_values, X_sample, expected_val, output_dir, model_name, sample_idx=0)

    top_features = get_top_features(shap_values, X_sample, top_n=10)
    print("\n[SHAP] Top 10 Churn Drivers:")
    print(top_features.to_string())

    # Save top features
    top_features.to_csv(output_dir / f"shap_top_features_{model_name.lower().replace(' ', '_')}.csv",
                         index=False)

    return shap_values, top_features


if __name__ == "__main__":
    import joblib
    base       = Path(__file__).parent.parent
    model      = joblib.load(base / "models" / "best_model.pkl")
    X_test     = pd.read_csv(base / "data" / "X_test.csv")
    run_explainability(
        model      = model,
        model_name = "Best Model",
        X_test     = X_test,
        output_dir = str(base / "images")
    )
