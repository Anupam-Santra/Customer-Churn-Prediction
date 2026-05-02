# ============================================================
# src/dashboard.py
# Interactive Plotly Dash Dashboard
# Run: python src/dashboard.py
# Then open: http://127.0.0.1:8050
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

# ── Paths ────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
DATA_DIR    = BASE / "data"
MODELS_DIR  = BASE / "models"

# ── Load Data & Model ────────────────────────────────────────
def load_assets():
    raw_df      = pd.read_csv(DATA_DIR / "customer_churn_raw.csv")
    eng_df      = pd.read_csv(DATA_DIR / "customer_churn_engineered.csv") \
                  if (DATA_DIR / "customer_churn_engineered.csv").exists() else raw_df
    X_test      = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test      = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    summary     = pd.read_csv(MODELS_DIR / "model_results_summary.csv", index_col=0) \
                  if (MODELS_DIR / "model_results_summary.csv").exists() else pd.DataFrame()
    return raw_df, eng_df, X_test, y_test, summary

raw_df, eng_df, X_test, y_test, model_summary = load_assets()

# Churn rate stats
churn_rate   = raw_df["churn"].mean() * 100
total_cust   = len(raw_df)
churned      = raw_df["churn"].sum()
retained     = total_cust - churned

# ── App Init ────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Churn Prediction Dashboard"
)
app.config.suppress_callback_exceptions = True

# ── Color scheme ─────────────────────────────────────────────
COLORS = {
    "primary"  : "#2C3E50",
    "churn"    : "#E74C3C",
    "retain"   : "#2ECC71",
    "warning"  : "#F39C12",
    "accent"   : "#3498DB",
    "bg"       : "#F8F9FA",
    "card_bg"  : "#FFFFFF",
}

# ── KPI Card Helper ──────────────────────────────────────────
def kpi_card(title, value, subtitle, color, icon):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={"fontSize": "2rem"}),
                html.Div([
                    html.H3(value, className="mb-0 fw-bold", style={"color": color}),
                    html.P(title, className="mb-0 text-muted small"),
                    html.P(subtitle, className="mb-0", style={"fontSize": "0.75rem", "color": color}),
                ], className="ms-3")
            ], className="d-flex align-items-center")
        ])
    ], className="shadow-sm border-0 h-100")


# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════
app.layout = dbc.Container([

    # ── Header ──────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.H1("🔮 Customer Churn Prediction Dashboard",
                    className="text-white fw-bold mb-0"),
            html.P("Advanced ML Analytics | Telecom Industry | Option C (Full Stack)",
                    className="text-white-50 mt-1"),
        ])
    ], className="bg-dark py-4 px-4 mb-4 rounded-3"),

    # ── KPI Row ──────────────────────────────────────────────
    dbc.Row([
        dbc.Col(kpi_card("Total Customers", f"{total_cust:,}", "Full dataset",
                          COLORS["accent"], "👥"), md=3),
        dbc.Col(kpi_card("Churned", f"{churned:,}", f"{churn_rate:.1f}% churn rate",
                          COLORS["churn"], "📉"), md=3),
        dbc.Col(kpi_card("Retained", f"{retained:,}", f"{100-churn_rate:.1f}% retention",
                          COLORS["retain"], "📈"), md=3),
        dbc.Col(kpi_card("Avg Monthly Charge", f"${raw_df['monthly_charges'].mean():.2f}",
                          "Per customer", COLORS["warning"], "💰"), md=3),
    ], className="mb-4 g-3"),

    # ── Tabs ─────────────────────────────────────────────────
    dbc.Tabs([

        # ── Tab 1: EDA ──────────────────────────────────────
        dbc.Tab(label="📊 Exploratory Analysis", tab_id="tab-eda", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Churn Distribution"),
                        dbc.CardBody(dcc.Graph(id="churn-pie"))
                    ], className="shadow-sm border-0")
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Select Feature to Explore"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="feature-dropdown",
                                options=[{"label": c.replace("_", " ").title(), "value": c}
                                         for c in ["tenure", "monthly_charges", "total_charges",
                                                    "age", "num_services", "engagement_score"]],
                                value="monthly_charges", clearable=False
                            ),
                            dcc.Graph(id="feature-hist")
                        ])
                    ], className="shadow-sm border-0")
                ], md=8),
            ], className="mt-3 g-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Churn Rate by Category"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="cat-dropdown",
                                options=[{"label": c.replace("_", " ").title(), "value": c}
                                         for c in ["contract_type", "internet_service",
                                                    "payment_method", "tech_support", "gender"]],
                                value="contract_type", clearable=False
                            ),
                            dcc.Graph(id="cat-churn-bar")
                        ])
                    ], className="shadow-sm border-0")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Tenure vs Monthly Charges"),
                        dbc.CardBody(dcc.Graph(id="scatter-plot"))
                    ], className="shadow-sm border-0")
                ], md=6),
            ], className="mt-3 g-3"),
        ]),

        # ── Tab 2: Model Performance ─────────────────────────
        dbc.Tab(label="🏆 Model Performance", tab_id="tab-models", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Leaderboard"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id="model-table",
                                columns=[{"name": c.replace("_", " ").title(), "id": c}
                                         for c in model_summary.reset_index().columns],
                                data=model_summary.reset_index().round(4).to_dict("records"),
                                style_cell={"textAlign": "center", "fontFamily": "Arial",
                                            "fontSize": "13px", "padding": "8px"},
                                style_header={"backgroundColor": COLORS["primary"],
                                              "color": "white", "fontWeight": "bold"},
                                style_data_conditional=[{
                                    "if": {"row_index": 0},
                                    "backgroundColor": "#D5F5E3",
                                    "fontWeight": "bold"
                                }],
                            )
                        ])
                    ], className="shadow-sm border-0")
                ], md=12),
            ], className="mt-3 g-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Metric Comparison (Bar Chart)"),
                        dbc.CardBody(dcc.Graph(id="metrics-bar"))
                    ], className="shadow-sm border-0")
                ], md=12),
            ], className="mt-3 g-3"),
        ]),

        # ── Tab 3: Business Insights ─────────────────────────
        dbc.Tab(label="💼 Business Insights", tab_id="tab-biz", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Revenue Impact Analysis"),
                        dbc.CardBody(dcc.Graph(id="revenue-chart"))
                    ], className="shadow-sm border-0")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Segmentation"),
                        dbc.CardBody(dcc.Graph(id="risk-donut"))
                    ], className="shadow-sm border-0")
                ], md=6),
            ], className="mt-3 g-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Retention Strategy Playbook"),
                        dbc.CardBody(html.Div(id="strategy-cards"))
                    ], className="shadow-sm border-0")
                ])
            ], className="mt-3"),
        ]),

        # ── Tab 4: Live Prediction ────────────────────────────
        dbc.Tab(label="🔮 Live Prediction", tab_id="tab-predict", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Enter Customer Details"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Tenure (months)"),
                                    dbc.Input(id="inp-tenure", type="number", value=6, min=1, max=72),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Monthly Charges ($)"),
                                    dbc.Input(id="inp-monthly", type="number", value=75.0, step=0.01),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Contract Type"),
                                    dcc.Dropdown(
                                        id="inp-contract",
                                        options=[
                                            {"label": "Month-to-month", "value": "Month-to-month"},
                                            {"label": "One year",        "value": "One year"},
                                            {"label": "Two year",        "value": "Two year"},
                                        ], value="Month-to-month", clearable=False
                                    ),
                                ], md=4),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Internet Service"),
                                    dcc.Dropdown(
                                        id="inp-internet",
                                        options=[
                                            {"label": "Fiber optic", "value": "Fiber optic"},
                                            {"label": "DSL",         "value": "DSL"},
                                            {"label": "No",          "value": "No"},
                                        ], value="Fiber optic", clearable=False
                                    ),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Tech Support"),
                                    dcc.Dropdown(
                                        id="inp-techsupport",
                                        options=[
                                            {"label": "Yes", "value": "Yes"},
                                            {"label": "No",  "value": "No"},
                                        ], value="No", clearable=False
                                    ),
                                ], md=4),
                                dbc.Col([
                                    dbc.Label("Payment Method"),
                                    dcc.Dropdown(
                                        id="inp-payment",
                                        options=[
                                            {"label": "Electronic check",          "value": "Electronic check"},
                                            {"label": "Mailed check",              "value": "Mailed check"},
                                            {"label": "Bank transfer (automatic)", "value": "Bank transfer (automatic)"},
                                            {"label": "Credit card (automatic)",   "value": "Credit card (automatic)"},
                                        ], value="Electronic check", clearable=False
                                    ),
                                ], md=4),
                            ], className="mb-3"),
                            dbc.Button("🔮 Predict Churn", id="predict-btn",
                                        color="danger", className="w-100 fw-bold"),
                        ])
                    ], className="shadow-sm border-0")
                ], md=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Prediction Result"),
                        dbc.CardBody(html.Div(id="prediction-output",
                                               style={"minHeight": "200px"}))
                    ], className="shadow-sm border-0 h-100")
                ], md=6),
            ], className="mt-3 g-3"),
        ]),

    ], id="main-tabs", active_tab="tab-eda"),

    html.Hr(),
    html.P("Customer Churn Prediction Model | Option C Advanced | Built for Placements & Internships",
            className="text-center text-muted small"),

], fluid=True, style={"backgroundColor": COLORS["bg"], "minHeight": "100vh"})


# ══════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════

@app.callback(Output("churn-pie", "figure"), Input("main-tabs", "active_tab"))
def update_pie(_):
    counts = raw_df["churn"].value_counts()
    fig = px.pie(
        names=["Retained", "Churned"], values=[counts.get(0, 0), counts.get(1, 0)],
        color_discrete_sequence=[COLORS["retain"], COLORS["churn"]],
        hole=0.45
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=10, b=10), showlegend=True,
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


@app.callback(Output("feature-hist", "figure"), Input("feature-dropdown", "value"))
def update_hist(feature):
    if feature not in eng_df.columns:
        return go.Figure()
    fig = px.histogram(
        eng_df, x=feature, color="churn",
        color_discrete_map={0: COLORS["retain"], 1: COLORS["churn"]},
        barmode="overlay", nbins=40, opacity=0.75,
        labels={"churn": "Churn", feature: feature.replace("_", " ").title()},
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=10))
    return fig


@app.callback(Output("cat-churn-bar", "figure"), Input("cat-dropdown", "value"))
def update_cat(col):
    if col not in raw_df.columns:
        return go.Figure()
    churn_rate_by_cat = (raw_df.groupby(col)["churn"].mean() * 100).reset_index()
    churn_rate_by_cat.columns = [col, "churn_rate"]
    fig = px.bar(churn_rate_by_cat, x=col, y="churn_rate",
                  color="churn_rate", color_continuous_scale="RdYlGn_r",
                  labels={"churn_rate": "Churn Rate (%)", col: col.replace("_", " ").title()})
    fig.add_hline(y=raw_df["churn"].mean() * 100, line_dash="dash",
                   line_color="navy", annotation_text="Avg")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=10), coloraxis_showscale=False)
    return fig


@app.callback(Output("scatter-plot", "figure"), Input("main-tabs", "active_tab"))
def update_scatter(_):
    sample = raw_df.sample(min(2000, len(raw_df)), random_state=42)
    fig = px.scatter(
        sample, x="tenure", y="monthly_charges", color=sample["churn"].map({0: "Retained", 1: "Churned"}),
        color_discrete_map={"Retained": COLORS["retain"], "Churned": COLORS["churn"]},
        opacity=0.5, size_max=6,
        labels={"tenure": "Tenure (months)", "monthly_charges": "Monthly Charges ($)"}
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=10), legend_title="")
    return fig


@app.callback(Output("metrics-bar", "figure"), Input("main-tabs", "active_tab"))
def update_metrics_bar(_):
    if model_summary.empty:
        return go.Figure()
    metrics_to_show = [c for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
                        if c in model_summary.columns]
    df_melt = model_summary[metrics_to_show].reset_index().melt(
        id_vars="index", var_name="Metric", value_name="Score"
    )
    df_melt.rename(columns={"index": "Model"}, inplace=True)
    fig = px.bar(df_melt, x="Model", y="Score", color="Metric", barmode="group",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(yaxis_range=[0, 1.05], paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
    return fig


@app.callback(Output("revenue-chart", "figure"), Input("main-tabs", "active_tab"))
def update_revenue(_):
    arpu             = raw_df["monthly_charges"].mean()
    churners         = int(raw_df["churn"].sum())
    high_risk        = int(churners * 0.55)
    revenue_at_risk  = churners * arpu * 12
    campaign_cost    = high_risk * 50
    revenue_saved    = int(high_risk * 0.35) * arpu * 12
    net_savings      = revenue_saved - campaign_cost

    categories = ["Revenue at Risk", "Campaign Cost", "Revenue Saved", "Net Savings"]
    values     = [revenue_at_risk, campaign_cost, revenue_saved, net_savings]
    colors     = [COLORS["churn"], COLORS["warning"], COLORS["accent"], COLORS["retain"]]

    fig = go.Figure(go.Bar(x=categories, y=values, marker_color=colors,
                             text=[f"${v:,.0f}" for v in values], textposition="outside"))
    fig.update_layout(title="Financial Impact of Churn & Retention",
                       yaxis_title="USD ($)", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40))
    return fig


@app.callback(Output("risk-donut", "figure"), Input("main-tabs", "active_tab"))
def update_risk_donut(_):
    # Simulate risk segments from churn probabilities
    n          = len(raw_df)
    high_risk  = int(n * 0.20)
    med_risk   = int(n * 0.30)
    low_risk   = n - high_risk - med_risk
    fig = go.Figure(go.Pie(
        labels=["High Risk", "Medium Risk", "Low Risk"],
        values=[high_risk, med_risk, low_risk],
        hole=0.5,
        marker_colors=[COLORS["churn"], COLORS["warning"], COLORS["retain"]],
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
    return fig


@app.callback(Output("strategy-cards", "children"), Input("main-tabs", "active_tab"))
def update_strategy(_):
    playbook = {
        "🔴 High Risk": {
            "color"     : "danger",
            "strategies": [
                "📞 Immediate retention call from senior account manager",
                "💰 Offer 20% discount for 3 months if they upgrade contract",
                "🔒 Free add-ons: Tech Support + Online Security for 60 days",
                "📊 Personalised usage report showing value they're getting",
            ]
        },
        "🟡 Medium Risk": {
            "color"     : "warning",
            "strategies": [
                "📧 Personalised email campaign with loyalty reward",
                "💡 Highlight underutilised features in their current plan",
                "🎁 Offer one free month of premium streaming add-on",
            ]
        },
        "🟢 Low Risk": {
            "color"     : "success",
            "strategies": [
                "🌟 Enrol in loyalty rewards programme",
                "🎉 Send anniversary discount (1yr, 2yr milestone)",
                "📣 Invite to beta features and early access programme",
            ]
        }
    }
    cards = []
    for seg, info in playbook.items():
        cards.append(
            dbc.Card([
                dbc.CardHeader(html.H5(seg, className="mb-0")),
                dbc.CardBody([
                    html.Ul([html.Li(s) for s in info["strategies"]])
                ])
            ], color=info["color"], outline=True, className="mb-3")
        )
    return cards


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("inp-tenure",      "value"),
    State("inp-monthly",     "value"),
    State("inp-contract",    "value"),
    State("inp-internet",    "value"),
    State("inp-techsupport", "value"),
    State("inp-payment",     "value"),
    prevent_initial_call=True,
)
def live_predict(n_clicks, tenure, monthly, contract, internet, techsupport, payment):
    if not n_clicks:
        return ""
    try:
        from predictor import ChurnPredictor
        predictor = ChurnPredictor(str(MODELS_DIR))

        total_charges = (monthly or 65) * (tenure or 6)
        customer = {
            "customer_id"      : "LIVE001",
            "age"              : 35,
            "gender"           : "Male",
            "senior_citizen"   : 0,
            "has_partner"      : 0,
            "has_dependents"   : 0,
            "tenure"           : tenure or 6,
            "phone_service"    : 1,
            "multiple_lines"   : "No",
            "internet_service" : internet or "Fiber optic",
            "online_security"  : "No",
            "online_backup"    : "No",
            "device_protection": "No",
            "tech_support"     : techsupport or "No",
            "streaming_tv"     : "No",
            "streaming_movies" : "No",
            "contract_type"    : contract or "Month-to-month",
            "paperless_billing": 1,
            "payment_method"   : payment or "Electronic check",
            "monthly_charges"  : monthly or 65.0,
            "total_charges"    : total_charges,
        }
        result  = predictor.predict_single(customer)
        prob    = result["churn_probability"]
        risk    = result["risk_level"]
        pred    = result["churn_prediction"]

        color   = "danger" if risk == "High Risk" else ("warning" if risk == "Medium Risk" else "success")
        label   = "⚠️ WILL CHURN" if pred == 1 else "✅ WILL STAY"

        gauge_fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = prob * 100,
            title = {"text": "Churn Probability (%)"},
            gauge = {
                "axis"  : {"range": [0, 100]},
                "bar"   : {"color": COLORS["churn"] if prob > 0.5 else COLORS["retain"]},
                "steps" : [
                    {"range": [0,  30], "color": "#D5F5E3"},
                    {"range": [30, 60], "color": "#FDEBD0"},
                    {"range": [60, 100],"color": "#FADBD8"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
            }
        ))
        gauge_fig.update_layout(height=250, margin=dict(t=30, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)")

        from src.business_insights import generate_retention_strategy
        strategies = generate_retention_strategy(risk)

        return [
            dbc.Alert(f"{label} | {risk} | Probability: {prob:.2%}",
                       color=color, className="fw-bold text-center"),
            dcc.Graph(figure=gauge_fig),
            html.H6("Recommended Actions:"),
            html.Ul([html.Li(s) for s in strategies]),
        ]
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}\n\nMake sure you run main.py first to train the model.",
                          color="warning")


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚀 Churn Prediction Dashboard")
    print("  Open your browser at: http://127.0.0.1:8050")
    print("="*55 + "\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
