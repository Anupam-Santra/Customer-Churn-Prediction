# ============================================================
# src/visualization.py
# Full EDA + Model Evaluation Visualizations
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

# ── Global style ─────────────────────────────────────────────
PALETTE   = {"0": "#2ECC71", "1": "#E74C3C"}
BG_COLOR  = "#F8F9FA"
GRID_ALPHA = 0.3

def _save(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[VIZ] Saved → {out}")


# ══════════════════════════════════════════════════════════════
#  EDA PLOTS
# ══════════════════════════════════════════════════════════════

def plot_churn_distribution(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_COLOR)
    fig.suptitle("Customer Churn Distribution", fontsize=15, fontweight="bold")

    counts = df["churn"].value_counts()
    labels = ["Retained (0)", "Churned (1)"]
    colors = ["#2ECC71", "#E74C3C"]

    axes[0].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Churn Ratio", fontsize=12)

    axes[1].bar(labels, counts, color=colors, edgecolor="white", width=0.5)
    for i, v in enumerate(counts):
        axes[1].text(i, v + 50, str(v), ha="center", fontweight="bold")
    axes[1].set_title("Churn Count", fontsize=12)
    axes[1].set_ylabel("Number of Customers")
    axes[1].grid(axis="y", alpha=GRID_ALPHA)
    axes[1].set_facecolor(BG_COLOR)

    _save(fig, output_dir, "01_churn_distribution.png")


def plot_numerical_distributions(df: pd.DataFrame, output_dir: Path):
    # Only use columns that exist in the dataframe (raw or engineered)
    candidates = ["age", "tenure", "monthly_charges", "total_charges",
                  "num_services", "engagement_score"]
    num_cols = [c for c in candidates if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=BG_COLOR)
    fig.suptitle("Numerical Feature Distributions by Churn", fontsize=14, fontweight="bold")

    for ax, col in zip(axes.flat, num_cols):
        for label, color in [("0", "#2ECC71"), ("1", "#E74C3C")]:
            subset = df[df["churn"] == int(label)][col].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label=f"{'Retained' if label=='0' else 'Churned'}", edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=GRID_ALPHA)
        ax.set_facecolor(BG_COLOR)

    plt.tight_layout()
    _save(fig, output_dir, "02_numerical_distributions.png")


def plot_categorical_churn_rates(df: pd.DataFrame, output_dir: Path):
    cat_cols = ["contract_type", "internet_service", "payment_method",
                "gender", "tech_support", "online_security"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG_COLOR)
    fig.suptitle("Churn Rate by Categorical Features", fontsize=14, fontweight="bold")

    for ax, col in zip(axes.flat, cat_cols):
        churn_rate = df.groupby(col)["churn"].mean().sort_values(ascending=False) * 100
        bars = ax.bar(churn_rate.index, churn_rate.values,
                      color=plt.cm.RdYlGn_r(churn_rate.values / 100),
                      edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xticklabels(churn_rate.index, rotation=25, ha="right", fontsize=8)
        ax.axhline(df["churn"].mean() * 100, color="navy", linestyle="--",
                   linewidth=1.2, label="Avg churn rate")
        for bar, val in zip(bars, churn_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=GRID_ALPHA)
        ax.set_facecolor(BG_COLOR)

    plt.tight_layout()
    _save(fig, output_dir, "03_categorical_churn_rates.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["senior_citizen"], errors="ignore")
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11), facecolor=BG_COLOR)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, output_dir, "04_correlation_heatmap.png")


def plot_tenure_vs_charges(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    for label, color, name in [(0, "#2ECC71", "Retained"), (1, "#E74C3C", "Churned")]:
        subset = df[df["churn"] == label]
        ax.scatter(subset["tenure"], subset["monthly_charges"],
                   alpha=0.4, s=20, color=color, label=name)
    ax.set_xlabel("Tenure (months)", fontsize=12)
    ax.set_ylabel("Monthly Charges ($)", fontsize=12)
    ax.set_title("Tenure vs Monthly Charges by Churn Status", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=GRID_ALPHA)
    ax.set_facecolor(BG_COLOR)
    _save(fig, output_dir, "05_tenure_vs_charges.png")


# ══════════════════════════════════════════════════════════════
#  MODEL EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrices(results: dict, y_test, output_dir: Path):
    n      = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), facecolor=BG_COLOR)
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm   = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10, fontweight="bold")

    plt.tight_layout()
    _save(fig, output_dir, "06_confusion_matrices.png")


def plot_roc_curves(results: dict, y_test, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    colors  = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=GRID_ALPHA)
    ax.set_facecolor(BG_COLOR)
    _save(fig, output_dir, "07_roc_curves.png")


def plot_precision_recall_curves(results: dict, y_test, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    colors  = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        pr_auc       = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AUC = {pr_auc:.4f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=GRID_ALPHA)
    ax.set_facecolor(BG_COLOR)
    _save(fig, output_dir, "08_precision_recall_curves.png")


def plot_model_comparison_bar(results: dict, output_dir: Path):
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    df      = pd.DataFrame({name: {m: res[m] for m in metrics}
                             for name, res in results.items()}).T

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_COLOR)
    x       = np.arange(len(df))
    width   = 0.15
    colors  = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics) / 2) * width + width / 2
        bars   = ax.bar(x + offset, df[metric], width, label=metric.replace("_", " ").title(),
                        color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_facecolor(BG_COLOR)
    plt.tight_layout()
    _save(fig, output_dir, "09_model_comparison.png")


def plot_feature_importance_rf(model, feature_names: list, output_dir: Path):
    """Random Forest built-in feature importance."""
    if not hasattr(model, "feature_importances_"):
        return
    imp    = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(15)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(imp)))

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    imp.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Feature Importances — Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=GRID_ALPHA)
    ax.set_facecolor(BG_COLOR)
    plt.tight_layout()
    _save(fig, output_dir, "10_feature_importance_rf.png")


def run_all_eda(df: pd.DataFrame, output_dir: str):
    """Run all EDA plots."""
    out = Path(output_dir)
    print("\n[VIZ] Generating EDA plots...")
    plot_churn_distribution(df, out)
    plot_numerical_distributions(df, out)
    plot_categorical_churn_rates(df, out)
    plot_correlation_heatmap(df, out)
    plot_tenure_vs_charges(df, out)
    print("[VIZ] EDA plots complete.")


def run_all_evaluation(results: dict, y_test, rf_model, feature_names: list, output_dir: str):
    """Run all model evaluation plots."""
    out = Path(output_dir)
    print("\n[VIZ] Generating model evaluation plots...")
    plot_confusion_matrices(results, y_test, out)
    plot_roc_curves(results, y_test, out)
    plot_precision_recall_curves(results, y_test, out)
    plot_model_comparison_bar(results, out)
    if rf_model:
        plot_feature_importance_rf(rf_model, feature_names, out)
    print("[VIZ] Evaluation plots complete.")
