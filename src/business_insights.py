# ============================================================
# src/business_insights.py
# Generate Business Insights Report from Model Results
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def generate_retention_strategy(risk_level: str) -> list:
    """Return actionable retention strategies by risk level."""
    strategies = {
        "High Risk": [
            "📞 Immediate outreach via phone call from retention team",
            "💰 Offer personalised discount (15-25% off for 3 months)",
            "🔒 Incentivise upgrade to annual/2-year contract",
            "🎁 Provide free premium add-on for 2 months (tech support, security)",
            "📊 Assign dedicated account manager",
        ],
        "Medium Risk": [
            "📧 Send targeted email with loyalty rewards",
            "💡 Highlight unused service benefits they're paying for",
            "🔄 Offer contract upgrade with minor discount",
            "📱 Proactive SMS check-in survey",
        ],
        "Low Risk": [
            "🌟 Enrol in loyalty/rewards programme",
            "📣 Invite to beta features / early access",
            "🎉 Acknowledge tenure milestones (1yr, 2yr rewards)",
        ]
    }
    return strategies.get(risk_level, [])


def calculate_business_metrics(predictions: pd.DataFrame,
                                 avg_revenue_per_customer: float = 65.0,
                                 retention_cost: float = 50.0,
                                 retention_success_rate: float = 0.35) -> dict:
    """
    Estimate financial impact of churn and retention campaign.

    Parameters:
        predictions              : Output from ChurnPredictor.predict()
        avg_revenue_per_customer : Monthly ARPU ($)
        retention_cost           : Cost to retain one customer ($)
        retention_success_rate   : % of high-risk customers we can retain
    """
    total         = len(predictions)
    churners      = int(predictions["churn_prediction"].sum())
    high_risk     = int((predictions["risk_level"] == "High Risk").sum())
    medium_risk   = int((predictions["risk_level"] == "Medium Risk").sum())
    low_risk      = int((predictions["risk_level"] == "Low Risk").sum())

    # Revenue at risk (annual)
    annual_revenue_at_risk = churners * avg_revenue_per_customer * 12

    # Savings if we successfully retain high-risk customers
    retained_customers     = round(high_risk * retention_success_rate)
    revenue_saved          = retained_customers * avg_revenue_per_customer * 12
    campaign_cost          = high_risk * retention_cost
    net_savings            = revenue_saved - campaign_cost
    roi_pct                = (net_savings / max(campaign_cost, 1)) * 100

    return {
        "total_customers"         : total,
        "predicted_churners"      : churners,
        "churn_rate_pct"          : round(churners / total * 100, 2),
        "high_risk_customers"     : high_risk,
        "medium_risk_customers"   : medium_risk,
        "low_risk_customers"      : low_risk,
        "annual_revenue_at_risk"  : round(annual_revenue_at_risk, 2),
        "retained_via_campaign"   : retained_customers,
        "revenue_saved"           : round(revenue_saved, 2),
        "campaign_cost"           : round(campaign_cost, 2),
        "net_savings"             : round(net_savings, 2),
        "roi_pct"                 : round(roi_pct, 2),
        "avg_churn_probability"   : round(float(predictions["churn_probability"].mean()), 4),
    }


def plot_business_dashboard(metrics: dict, top_features: pd.DataFrame, output_dir: Path):
    """Generate a 2×2 business summary dashboard."""
    fig = plt.figure(figsize=(16, 12), facecolor="#F0F4F8")
    fig.suptitle("Customer Churn — Business Intelligence Dashboard",
                  fontsize=18, fontweight="bold", y=0.98, color="#1A202C")

    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Risk Segment Donut ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sizes  = [metrics["high_risk_customers"],
              metrics["medium_risk_customers"],
              metrics["low_risk_customers"]]
    labels = ["High Risk", "Medium Risk", "Low Risk"]
    colors = ["#E74C3C", "#F39C12", "#2ECC71"]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2.5},
        textprops={"fontsize": 9}
    )
    # Donut hole
    centre_circle = plt.Circle((0, 0), 0.60, fc="#F0F4F8")
    ax1.add_patch(centre_circle)
    ax1.text(0, 0, f"{metrics['churn_rate_pct']}%\nChurn", ha="center",
              va="center", fontsize=12, fontweight="bold", color="#E74C3C")
    ax1.set_title("Risk Segmentation", fontsize=12, fontweight="bold")

    # ── 2. Financial Summary ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    fin_labels = ["Revenue\nat Risk", "Campaign\nCost", "Revenue\nSaved", "Net\nSavings"]
    fin_values = [
        metrics["annual_revenue_at_risk"],
        metrics["campaign_cost"],
        metrics["revenue_saved"],
        metrics["net_savings"]
    ]
    fin_colors = ["#E74C3C", "#F39C12", "#3498DB", "#2ECC71"]
    bars = ax2.bar(fin_labels, fin_values, color=fin_colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, fin_values):
        sign = "+" if val >= 0 else ""
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                  f"${sign}{val:,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("Financial Impact ($)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Amount (USD)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_facecolor("#F0F4F8")

    # ── 3. KPI Cards ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    kpis = [
        ("Total Customers",    f"{metrics['total_customers']:,}",   "#3498DB"),
        ("Predicted Churners", f"{metrics['predicted_churners']:,}", "#E74C3C"),
        ("High Risk",          f"{metrics['high_risk_customers']:,}","#E67E22"),
        ("ROI of Campaign",    f"{metrics['roi_pct']:.1f}%",         "#27AE60"),
        ("Avg Churn Prob",     f"{metrics['avg_churn_probability']:.2%}", "#8E44AD"),
    ]
    for i, (label, value, color) in enumerate(kpis):
        y = 0.92 - i * 0.19
        ax3.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.08), 0.96, 0.15,
            boxstyle="round,pad=0.01", linewidth=1.5,
            edgecolor=color, facecolor=color + "22"
        ))
        ax3.text(0.10, y, label,  fontsize=9,  color="#2C3E50", va="center")
        ax3.text(0.90, y, value,  fontsize=12, color=color, va="center",
                  ha="right", fontweight="bold")
    ax3.set_title("Key Performance Indicators", fontsize=12, fontweight="bold")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

    # ── 4. Top Churn Drivers ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    if top_features is not None and not top_features.empty:
        feats  = top_features["feature"].str.replace("_", " ").str.title().head(10)
        vals   = top_features["mean_shap_abs"].head(10)
        colors_bar = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(feats)))
        ax4.barh(feats[::-1], vals[::-1], color=colors_bar[::-1], edgecolor="white")
        ax4.set_title("Top 10 Churn Drivers (SHAP)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Mean |SHAP Value|")
        ax4.grid(axis="x", alpha=0.3)
    ax4.set_facecolor("#F0F4F8")

    # ── 5. Retention Strategies ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.set_title("Retention Playbook", fontsize=12, fontweight="bold")
    strategies = {
        "🔴 High Risk"  : generate_retention_strategy("High Risk")[:2],
        "🟡 Medium Risk": generate_retention_strategy("Medium Risk")[:2],
        "🟢 Low Risk"   : generate_retention_strategy("Low Risk")[:1],
    }
    y = 0.95
    for seg, tips in strategies.items():
        ax5.text(0.02, y, seg, fontsize=9, fontweight="bold", color="#2C3E50", va="top")
        y -= 0.06
        for tip in tips:
            ax5.text(0.04, y, tip, fontsize=7.5, color="#555", va="top", wrap=True)
            y -= 0.10
        y -= 0.02
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)

    out = output_dir / "11_business_dashboard.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#F0F4F8")
    plt.close(fig)
    print(f"[BIZ] Business dashboard saved → {out}")


def print_business_report(metrics: dict):
    """Print formatted business report to console."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  BUSINESS INTELLIGENCE REPORT")
    print(sep)
    print(f"  Total Customers Analysed  : {metrics['total_customers']:>10,}")
    print(f"  Predicted to Churn        : {metrics['predicted_churners']:>10,}  ({metrics['churn_rate_pct']}%)")
    print(f"  High Risk Customers       : {metrics['high_risk_customers']:>10,}")
    print(f"  Medium Risk Customers     : {metrics['medium_risk_customers']:>10,}")
    print(f"  Low Risk Customers        : {metrics['low_risk_customers']:>10,}")
    print(f"\n  Annual Revenue at Risk    : ${metrics['annual_revenue_at_risk']:>12,.2f}")
    print(f"  Retention Campaign Cost   : ${metrics['campaign_cost']:>12,.2f}")
    print(f"  Revenue Saved (est.)      : ${metrics['revenue_saved']:>12,.2f}")
    print(f"  Net Savings               : ${metrics['net_savings']:>12,.2f}")
    print(f"  Campaign ROI              : {metrics['roi_pct']:>11.1f}%")
    print(sep + "\n")


if __name__ == "__main__":
    from predictor import ChurnPredictor, SAMPLE_CUSTOMERS
    base      = Path(__file__).parent.parent
    predictor = ChurnPredictor(str(base / "models"))
    df_demo   = pd.DataFrame(SAMPLE_CUSTOMERS)
    results   = predictor.predict(df_demo)
    metrics   = calculate_business_metrics(results)
    print_business_report(metrics)
    plot_business_dashboard(metrics, None, base / "images")
