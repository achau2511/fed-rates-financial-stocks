import os
import pickle
import psycopg2
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fed Rates & Financial Stocks",
    page_icon="📈",
    layout="wide",
)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "fed_rates_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "best_model.pkl")

TICKERS = ["JPM", "GS", "BAC", "WFC", "C", "MS"]

CAT_MAPPINGS = {
    "direction":         {"hike": 1, "cut": -1, "hold": 0},
    "rate_level_regime": {"low": 0, "mid": 1, "high": 2},
}

# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_fed_rates():
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql(
            "SELECT date, rate, change_bp FROM fed_rates ORDER BY date",
            conn, parse_dates=["date"]
        )

@st.cache_data
def load_fomc_meetings():
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql(
            "SELECT date, rate, change_bp FROM fomc_meetings ORDER BY date",
            conn, parse_dates=["date"]
        )

@st.cache_data
def load_stock_prices():
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql(
            "SELECT date, ticker, close FROM stock_prices ORDER BY date",
            conn, parse_dates=["date"]
        )

@st.cache_data
def load_features():
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql(
            "SELECT * FROM features ORDER BY fomc_date, ticker",
            conn, parse_dates=["fomc_date"]
        )

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def compute_time_split_metrics():
    """
    Honest evaluation: chronological 80/20 split on FOMC date.
    Trains a fresh copy of the saved model pipeline on the train period only,
    then evaluates on the held-out later period. Returns None if the model
    or features aren't available.
    """
    payload = load_model()
    if payload is None:
        return None

    model_template = payload["model"]
    feature_cols   = payload["feature_cols"]
    cat_mappings   = payload["cat_mappings"]

    features = load_features()
    if features.empty or "fomc_date" not in features.columns:
        return None

    df = features.copy()
    # Keep the original (string) categoricals around for display
    string_cat_cols = list(cat_mappings.keys())
    df_str = df[string_cat_cols].add_suffix("_label")
    for col, mapping in cat_mappings.items():
        df[col] = df[col].map(mapping)
    df = pd.concat([df, df_str], axis=1)
    df = df.dropna(subset=feature_cols + ["outperformed", "fomc_date"])
    df = df.sort_values("fomc_date").reset_index(drop=True)

    unique_dates = sorted(df["fomc_date"].unique())
    if len(unique_dates) < 5:
        return None

    # Split on unique meeting dates so all tickers for a given meeting stay together
    split_idx  = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]
    train_df   = df[df["fomc_date"] < split_date]
    test_df    = df[df["fomc_date"] >= split_date]

    if train_df.empty or test_df.empty:
        return None

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["outperformed"].astype(int)
    X_test  = test_df[feature_cols].astype(float)
    y_test  = test_df["outperformed"].astype(int)

    fresh_model = clone(model_template)
    fresh_model.fit(X_train, y_train)

    y_pred  = fresh_model.predict(X_test)
    y_proba = fresh_model.predict_proba(X_test)[:, 1]

    # Baseline = always predict the majority class from the train set
    majority_class = int(y_train.value_counts().idxmax())
    baseline_acc   = float((y_test == majority_class).mean())

    test_with_pred = test_df.copy()
    test_with_pred["predicted"] = y_pred
    test_with_pred["proba"]     = y_proba

    return {
        "accuracy":           float(accuracy_score(y_test, y_pred)),
        "precision":          float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":             float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":                 float(f1_score(y_test, y_pred, zero_division=0)),
        "baseline_accuracy":  baseline_acc,
        "majority_class":     majority_class,
        "confusion_matrix":   confusion_matrix(y_test, y_pred),
        "n_train":            int(len(train_df)),
        "n_test":             int(len(test_df)),
        "train_start":        train_df["fomc_date"].min(),
        "train_end":          train_df["fomc_date"].max(),
        "test_start":         test_df["fomc_date"].min(),
        "test_end":           test_df["fomc_date"].max(),
        "positive_rate_test": float(y_test.mean()),
        "test_predictions":   test_with_pred,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def direction_label(bp):
    if bp > 0:   return "🔺 Hike"
    if bp < 0:   return "🔻 Cut"
    return "⏸ Hold"

def direction_color(bp):
    if bp > 0:   return "#EF4444"
    if bp < 0:   return "#22C55E"
    return "#94A3B8"


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📈 Fed Rates Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Rate History", "📉 Stock Performance", "🤖 Model Predictions", "🔮 Scenario Testing"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: FRED API + Yahoo Finance\nModel: Gradient Boosting Classifier")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("Fed Interest Rate Changes & Financial Sector Stock Performance")
    st.markdown("*Can Federal Reserve interest rate decisions predict short-term stock outperformance?*")
    st.markdown("---")

    fed     = load_fed_rates()
    fomc    = load_fomc_meetings()
    stocks  = load_stock_prices()
    features = load_features()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily Rate Observations", f"{len(fed):,}")
    col2.metric("FOMC Meetings", f"{len(fomc):,}")
    col3.metric("Stock Price Rows", f"{len(stocks):,}")
    col4.metric("FOMC Event-Stock Pairs", f"{len(features):,}")

    st.markdown("---")

    # Mini rate chart
    st.subheader("Federal Funds Rate — 2000 to Present")
    fig = px.area(
        fed, x="date", y="rate",
        labels={"rate": "Rate (%)", "date": ""},
        color_discrete_sequence=["#1E2761"],
    )
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The Fed Funds Rate has cycled through three major regimes since 2000: near-zero rates post-2008, a rapid hiking campaign in 2022–2023, and gradual cuts since. Each regime shift had distinct effects on financial sector stocks.")

    # Recent FOMC meetings table
    st.subheader("Recent FOMC Meetings")
    recent = fomc.tail(8).sort_values("date", ascending=False).copy()
    recent["date"]      = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
    recent["Direction"] = recent["change_bp"].apply(direction_label)
    recent["change_bp"] = recent["change_bp"].apply(lambda x: f"{x:+.0f} bps")
    recent["rate"]      = recent["rate"].apply(lambda x: f"{x:.2f}%")
    recent.columns      = ["Date", "Rate", "Change", "Direction"]
    st.dataframe(recent.set_index("Date"), use_container_width=True)
    st.caption("Rate decisions often come in streaks — the Fed rarely reverses course after a single meeting. Sustained hold periods typically signal uncertainty rather than stability.")

    st.markdown("---")

    # Methodology
    with st.expander("📚 Methodology — how the analysis is built"):
        st.markdown("""
        **Definitions used throughout the dashboard:**

        - **FOMC meeting** — A day on which the Fed funds target rate changed (derived from the FRED `DFEDTARU` series since 2000).
        - **30-day return** — Price return of a stock over the 30 trading days *after* the FOMC meeting.
        - **30-day return vs SPY** — `stock_30d_return − spy_30d_return`. Positive means the stock beat the broad market.
        - **Outperform / Underperform SPY** — Binary label: `1` if the stock's 30-day return is greater than SPY's, else `0`.
        - **Features** — Pre-meeting stock momentum (10d, 30d), 30-day annualized volatility, relative returns vs SPY (10d, 30d), the rate decision direction (hike/cut/hold), the change size in basis points (bps), and the rate-level regime (low <2%, mid 2–4%, high >4%).
        - **Target** — Did the stock outperform SPY over the next 30 trading days?
        - **Model** — Gradient Boosting Classifier. Evaluated using a **chronological 80/20 train/test split** on FOMC dates so the model is tested only on later events it never saw during training.

        **⚠️ Disclaimer.** This is exploratory research on a small event-level sample (~30 FOMC events × 6 stocks). Results may be unstable across market regimes and should **not** be interpreted as a trading system or investment advice.
        """)

    st.markdown("---")

    # Top / worst post-FOMC performers (uses existing fields only)
    st.subheader("Most Extreme Post-FOMC Outcomes")
    perf = features.copy()
    perf["rel_return_30d"] = perf["stock_return_30d"] - perf["spy_return_30d"]
    perf["fomc_date"]      = pd.to_datetime(perf["fomc_date"]).dt.date
    perf_cols = ["fomc_date", "ticker", "direction", "change_bp", "stock_return_30d", "spy_return_30d", "rel_return_30d"]

    def _format_perf(d):
        out = d[perf_cols].copy()
        out["change_bp"]        = out["change_bp"].apply(lambda x: f"{x:+.0f} bps")
        out["stock_return_30d"] = out["stock_return_30d"].apply(lambda x: f"{x*100:+.1f}%")
        out["spy_return_30d"]   = out["spy_return_30d"].apply(lambda x: f"{x*100:+.1f}%")
        out["rel_return_30d"]   = out["rel_return_30d"].apply(lambda x: f"{x*100:+.1f}%")
        out.columns = ["FOMC Date", "Ticker", "Direction", "Rate Change", "Stock 30d", "SPY 30d", "30d vs SPY"]
        return out.set_index("FOMC Date")

    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown("**🏆 Top 5 Outperformers vs SPY**")
        st.dataframe(_format_perf(perf.nlargest(5, "rel_return_30d")), use_container_width=True)
    with col_bot:
        st.markdown("**📉 Bottom 5 Underperformers vs SPY**")
        st.dataframe(_format_perf(perf.nsmallest(5, "rel_return_30d")), use_container_width=True)
    st.caption("30-day return vs SPY = stock 30-day return − SPY 30-day return. Positive = beat the market; negative = lagged the market.")

    st.markdown("---")

    # Win-rate by ticker — historical signal strength
    st.subheader("Historical Win Rate by Ticker (vs SPY, 30 days post-FOMC)")
    win = (
        perf.groupby("ticker")
            .agg(meetings=("outperformed", "count"),
                 wins=("outperformed", "sum"),
                 avg_rel_return=("rel_return_30d", "mean"))
            .assign(win_rate=lambda d: d["wins"] / d["meetings"])
            .sort_values("win_rate", ascending=False)
    )
    win_display = win.copy()
    win_display["win_rate"]       = win_display["win_rate"].apply(lambda x: f"{x*100:.1f}%")
    win_display["avg_rel_return"] = win_display["avg_rel_return"].apply(lambda x: f"{x*100:+.2f}%")
    win_display.columns = ["FOMC Meetings", "Outperformed SPY", "Avg 30d vs SPY", "Win Rate"]
    st.dataframe(win_display[["FOMC Meetings", "Outperformed SPY", "Win Rate", "Avg 30d vs SPY"]], use_container_width=True)
    st.caption("A win rate above 50% suggests the stock has historically had a positive edge over SPY in the 30 days following an FOMC meeting. Sample sizes are small — interpret with care. Win rates are calculated across all 31 historical FOMC meetings in the dataset, not just the held-out test set.")

    # Latest Signal — model's prediction for the most recent FOMC meeting
    latest_payload = load_model()
    if latest_payload is not None and not features.empty:
        latest_model        = latest_payload["model"]
        latest_feature_cols = latest_payload["feature_cols"]
        latest_cat_mappings = latest_payload["cat_mappings"]

        latest_date = features["fomc_date"].max()
        latest_rows = features[features["fomc_date"] == latest_date].copy()

        encoded = latest_rows.copy()
        for col, mapping in latest_cat_mappings.items():
            encoded[col] = encoded[col].map(mapping)
        encoded = encoded.dropna(subset=latest_feature_cols)

        if not encoded.empty:
            st.markdown("---")
            st.subheader("Latest Signal")
            st.markdown(f"**Most recent FOMC meeting in dataset:** {pd.to_datetime(latest_date).strftime('%Y-%m-%d')}")

            X_latest = encoded[latest_feature_cols].astype(float)
            probs    = latest_model.predict_proba(X_latest)[:, 1]

            signal = latest_rows.loc[encoded.index].copy()
            signal["P(Outperform)"]  = [f"{p:.1%}" for p in probs]
            signal["Classification"] = ["Outperform" if p >= 0.5 else "Underperform" for p in probs]
            signal["change_bp"]      = signal["change_bp"].apply(lambda x: f"{x:+.0f}")

            signal_cols = ["ticker", "direction", "change_bp", "P(Outperform)", "Classification"]
            st.dataframe(
                signal[signal_cols].rename(columns={
                    "ticker": "Ticker",
                    "direction": "Direction",
                    "change_bp": "Change (bps)",
                }).set_index("Ticker"),
                use_container_width=True,
            )
            st.caption("Based on the most recent FOMC meeting in the dataset. These are in-sample predictions — the model was trained on this event.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Rate History
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Rate History":
    st.title("Federal Reserve Rate History")

    fed  = load_fed_rates()
    fomc = load_fomc_meetings()

    # Date range filter
    min_date = fed["date"].min().date()
    max_date = fed["date"].max().date()
    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", value=pd.to_datetime("2008-01-01").date(), min_value=min_date, max_value=max_date)
    end   = col2.date_input("End date",   value=max_date, min_value=min_date, max_value=max_date)

    mask = (fed["date"].dt.date >= start) & (fed["date"].dt.date <= end)
    fed_filtered  = fed[mask]
    fomc_filtered = fomc[(fomc["date"].dt.date >= start) & (fomc["date"].dt.date <= end)]

    # Rate line chart with FOMC markers
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fed_filtered["date"], y=fed_filtered["rate"],
        mode="lines", name="Fed Funds Rate",
        line=dict(color="#1E2761", width=2)
    ))
    # FOMC hike markers
    hikes = fomc_filtered[fomc_filtered["change_bp"] > 0]
    cuts  = fomc_filtered[fomc_filtered["change_bp"] < 0]
    fig.add_trace(go.Scatter(
        x=hikes["date"], y=hikes["rate"],
        mode="markers", name="Rate Hike",
        marker=dict(color="#EF4444", size=10, symbol="triangle-up")
    ))
    fig.add_trace(go.Scatter(
        x=cuts["date"], y=cuts["rate"],
        mode="markers", name="Rate Cut",
        marker=dict(color="#22C55E", size=10, symbol="triangle-down")
    ))
    fig.update_layout(
        height=420,
        xaxis_title="", yaxis_title="Rate (%)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hike cycles (red) and cut cycles (green) reflect the Fed's response to inflation and recession risk respectively. Cuts tend to cluster at crisis inflection points — 2001, 2008, 2020.")

    # Change distribution
    st.subheader("Distribution of Rate Changes at FOMC Meetings")
    fig2 = px.histogram(
        fomc_filtered, x="change_bp", nbins=20,
        labels={"change_bp": "Change (basis points)"},
        color_discrete_sequence=["#1E2761"]
    )
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("25 bps moves are by far the most common — the Fed's preferred \"measured\" step. Larger moves (50–75 bps) are rare and typically signal emergency conditions or aggressive inflation-fighting.")

    # Raw FOMC table
    with st.expander("View all FOMC meetings in range"):
        show = fomc_filtered.copy().sort_values("date", ascending=False)
        show["date"]      = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
        show["Direction"] = show["change_bp"].apply(direction_label)
        st.dataframe(show, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Stock Performance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📉 Stock Performance":
    st.title("Stock Performance Around FOMC Meetings")

    features = load_features()
    stocks   = load_stock_prices()

    col1, col2 = st.columns(2)
    ticker  = col1.selectbox("Ticker", TICKERS)
    direction_filter = col2.selectbox("Rate direction", ["All", "Hike", "Cut", "Hold"])

    df = features[features["ticker"] == ticker].copy()
    if direction_filter != "All":
        df = df[df["direction"] == direction_filter.lower()]

    if df.empty:
        st.warning("No data for that combination.")
    else:
        # Win rate
        win_rate = df["outperformed"].mean() * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Meetings analysed", len(df))
        col2.metric("Outperformed SPY", f"{df['outperformed'].sum()} times")
        col3.metric("Win rate", f"{win_rate:.1f}%")

        # 30-day forward returns scatter
        st.subheader(f"{ticker} vs SPY — 30-Day Returns After Each FOMC Meeting")
        fig = go.Figure()
        colors = ["#22C55E" if o else "#EF4444" for o in df["outperformed"]]
        fig.add_trace(go.Scatter(
            x=df["fomc_date"], y=df["stock_return_30d"] * 100,
            mode="markers+lines", name=ticker,
            marker=dict(color=colors, size=10),
            line=dict(color="#94A3B8", width=1, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=df["fomc_date"], y=df["spy_return_30d"] * 100,
            mode="lines", name="SPY",
            line=dict(color="#1E2761", width=2)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            height=400, xaxis_title="FOMC Date",
            yaxis_title="30-Day Return (%)",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Green dots indicate {ticker} outperformed SPY in the 30 days after the FOMC meeting; red dots indicate underperformance. A win rate above 50% suggests the stock has a positive post-FOMC edge over the broader market.")

        # Stock price chart
        st.subheader(f"{ticker} Price History")
        ticker_prices = stocks[stocks["ticker"] == ticker]
        fig2 = px.line(ticker_prices, x="date", y="close",
                       labels={"close": "Price ($)", "date": ""},
                       color_discrete_sequence=["#1E2761"])
        # Mark FOMC dates
        fomc_prices = ticker_prices[ticker_prices["date"].isin(df["fomc_date"])]
        fig2.add_trace(go.Scatter(
            x=fomc_prices["date"], y=fomc_prices["close"],
            mode="markers", name="FOMC Date",
            marker=dict(color="#F5C842", size=8)
        ))
        fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Yellow markers highlight FOMC meeting dates on {ticker}'s price history — useful for spotting whether rate decisions aligned with major price inflection points.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Predictions
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Predictions":
    st.title("ML Model Predictions")

    payload = load_model()
    if payload is None:
        st.error("No trained model found. Run `python src/model.py` first.")
        st.stop()

    model        = payload["model"]
    model_name   = payload["model_name"]
    feature_cols = payload["feature_cols"]

    st.success(f"Loaded model: **{model_name}**")

    metrics = compute_time_split_metrics()
    if metrics is None:
        st.warning(
            "Could not run time-based evaluation — `fomc_date` was missing or the dataset was too small. "
            "Metrics shown below are exploratory only."
        )
        st.stop()

    # ── Evaluation period summary ────────────────────────────────────────────
    st.subheader("Out-of-Sample Evaluation")
    st.markdown(
        "Metrics below come from a **chronological 80/20 train/test split**: the model is fit on earlier "
        "FOMC events and evaluated on later events it never saw during training. This avoids the "
        "look-ahead leakage that produces inflated training accuracy."
    )

    period_col1, period_col2 = st.columns(2)
    with period_col1:
        st.markdown(
            f"**Train period:** {metrics['train_start']} → {metrics['train_end']}  \n"
            f"**Train samples:** {metrics['n_train']}"
        )
    with period_col2:
        st.markdown(
            f"**Test period:** {metrics['test_start']} → {metrics['test_end']}  \n"
            f"**Test samples:** {metrics['n_test']}  \n"
            f"**Positive class rate (test):** {metrics['positive_rate_test']*100:.1f}% "
            f"(share of test events where the stock beat SPY)"
        )

    # ── Diagnostics row ──────────────────────────────────────────────────────
    st.markdown("##### Test-Set Performance")
    m1, m2, m3 = st.columns(3)
    delta_vs_baseline = (metrics["accuracy"] - metrics["baseline_accuracy"]) * 100
    m1.metric(
        "Model Accuracy",
        f"{metrics['accuracy']*100:.1f}%",
        delta=f"{delta_vs_baseline:+.1f} pts vs baseline",
    )
    m2.metric("Baseline Accuracy", f"{metrics['baseline_accuracy']*100:.1f}%")
    m3.metric("F1 Score (Outperform)", f"{metrics['f1']:.3f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("Precision (Outperform)", f"{metrics['precision']:.3f}")
    m5.metric("Recall (Outperform)", f"{metrics['recall']:.3f}")
    m6.metric("Test Samples", f"{metrics['n_test']}")

    st.caption(
        "Baseline = always predicting the majority class from the training set. "
        "A model is only useful if it beats this. Precision/recall/F1 are reported "
        "for the positive class (stock outperformed SPY)."
    )

    # ── Confusion matrix ─────────────────────────────────────────────────────
    st.markdown("##### Confusion Matrix (Test Set)")
    cm = metrics["confusion_matrix"]
    # Pad in case the test set only contains one class
    if cm.shape != (2, 2):
        padded = np.zeros((2, 2), dtype=int)
        padded[:cm.shape[0], :cm.shape[1]] = cm
        cm = padded

    # Color-code per cell: TN/TP (correct) green, FP/FN (errors) red
    correctness = [[1, 0], [0, 1]]   # rows: actual, cols: predicted
    cm_fig = go.Figure(data=go.Heatmap(
        z=correctness,
        x=["Predicted Underperform", "Predicted Outperform"],
        y=["Actual Underperform", "Actual Outperform"],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        colorscale=[[0, "#EF4444"], [1, "#22C55E"]],
        zmin=0, zmax=1,
        showscale=False,
    ))
    cm_fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(cm_fig, use_container_width=True)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    st.caption(
        f"True negatives: {tn} · False positives: {fp} · "
        f"False negatives: {fn} · True positives: {tp}. "
        "On a small test set, even a few misclassifications swing the metrics meaningfully."
    )

    st.markdown("---")

    # ── Feature importance ───────────────────────────────────────────────────
    st.subheader("Feature Importance")
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.abs(clf.coef_[0])

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#CADCFC", "#1E2761"],
        labels={"Importance": "Importance Score"}
    )
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Pre-meeting momentum and volatility tend to rank as the strongest predictors — suggesting that "
        "how a stock was already moving matters more than the Fed's actual decision. "
        "**Note on ticker identity:** This model intentionally **excludes ticker** as a feature so the "
        "signal reflects macro/FOMC effects rather than firm-specific behavior. If a future version adds "
        "ticker dummies and they dominate, the model would mostly be learning persistent firm-level "
        "differences, not Fed effects."
    )

    st.markdown("---")

    # ── Test-set predictions table ───────────────────────────────────────────
    st.subheader("Predictions on the Test Set")
    st.caption(
        "These are predictions on the held-out chronological test period only — events the model "
        "did not see during training."
    )
    ticker_filter = st.selectbox("Filter by ticker", ["All"] + TICKERS)

    show = metrics["test_predictions"].copy()
    if ticker_filter != "All":
        show = show[show["ticker"] == ticker_filter]

    show["Predicted"] = show["predicted"].apply(lambda x: "✅ Outperform" if x == 1 else "❌ Underperform")
    show["Actual"]    = show["outperformed"].apply(lambda x: "✅ Outperform" if x == 1 else "❌ Underperform")
    show["Correct"]   = (show["predicted"] == show["outperformed"]).apply(lambda x: "✓" if x else "✗")
    show["P(Outperform)"] = show["proba"].apply(lambda x: f"{x:.1%}")
    show["change_bp"]     = show["change_bp"].apply(lambda x: f"{x:+.0f}")
    show["fomc_date"]     = pd.to_datetime(show["fomc_date"]).dt.strftime("%Y-%m-%d")

    cols_to_show = ["fomc_date", "ticker", "direction_label", "change_bp", "Predicted", "Actual", "Correct", "P(Outperform)"]
    st.dataframe(show[cols_to_show].rename(columns={
        "fomc_date": "FOMC Date", "ticker": "Ticker",
        "direction_label": "Direction", "change_bp": "Change (bps)"
    }).set_index("FOMC Date"), use_container_width=True)

    st.warning(
        "⚠️ **Exploratory model.** Trained on a small event-level sample. Results may be unstable across "
        "market regimes and should not be used as standalone investment advice."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Scenario Testing
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Scenario Testing":
    st.title("Scenario Testing")
    st.markdown("Adjust the inputs below to simulate a hypothetical FOMC meeting and see what the model predicts.")

    payload = load_model()
    if payload is None:
        st.error("No trained model found. Run `python src/model.py` first.")
        st.stop()

    model        = payload["model"]
    feature_cols = payload["feature_cols"]
    cat_mappings = payload["cat_mappings"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 FOMC Meeting Inputs")
        rate_before    = st.slider("Current Fed funds rate (%)", 0.0, 10.0, 5.25, step=0.25)
        direction      = st.selectbox("Rate decision", ["hike", "cut", "hold"])
        change_bp_abs  = st.slider("Size of change (basis points)", 0, 100, 25, step=25)
        change_bp      = change_bp_abs if direction == "hike" else (-change_bp_abs if direction == "cut" else 0)

        if rate_before < 2:
            regime = "low"
        elif rate_before <= 4:
            regime = "mid"
        else:
            regime = "high"
        st.caption(f"Rate regime: **{regime}**")

    with col2:
        st.subheader("📉 Stock Inputs (Pre-Meeting)")
        ticker         = st.selectbox("Ticker", TICKERS)
        pre_return_10d = st.slider("10-day return before meeting (%)", -20.0, 20.0, 0.0, step=0.5) / 100
        pre_return_30d = st.slider("30-day return before meeting (%)", -30.0, 30.0, 0.0, step=0.5) / 100
        pre_vol_30d    = st.slider("30-day annualised volatility (%)", 5.0, 80.0, 25.0, step=1.0) / 100
        pre_rel_10d    = st.slider("Relative return vs SPY — 10d (%)", -15.0, 15.0, 0.0, step=0.5) / 100
        pre_rel_30d    = st.slider("Relative return vs SPY — 30d (%)", -20.0, 20.0, 0.0, step=0.5) / 100

    st.markdown("---")

    # Build input vector
    input_data = {
        "rate_before":        rate_before,
        "change_bp":          change_bp,
        "abs_change_bp":      abs(change_bp),
        "pre_return_10d":     pre_return_10d,
        "pre_return_30d":     pre_return_30d,
        "pre_volatility_30d": pre_vol_30d,
        "pre_rel_return_10d": pre_rel_10d,
        "pre_rel_return_30d": pre_rel_30d,
        "direction":          cat_mappings["direction"][direction],
        "rate_level_regime":  cat_mappings["rate_level_regime"][regime],
    }
    X_input = pd.DataFrame([input_data])[feature_cols].astype(float)

    prob        = model.predict_proba(X_input)[0][1]
    prediction  = model.predict(X_input)[0]

    # Result display
    classification = "Outperform" if prob >= 0.5 else "Underperform"
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if classification == "Outperform":
            st.success(f"### ✅ Classification: {ticker} likely to OUTPERFORM SPY")
        else:
            st.error(f"### ❌ Classification: {ticker} likely to UNDERPERFORM SPY")
        st.markdown(f"**Predicted probability of outperforming SPY: {prob:.1%}**")
        st.progress(prob)
        st.caption(
            "This probability reflects the model's estimated chance that the selected stock outperforms "
            "SPY over the 30 trading days following the hypothetical FOMC meeting. A probability ≥ 50% "
            "yields an *Outperform* classification; below 50% yields *Underperform*. This is a model-"
            "estimated probability, not a calibrated confidence."
        )

    # Probability gauge
    st.markdown("---")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Outperformance Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1E2761"},
            "steps": [
                {"range": [0, 40],  "color": "#7F1D1D"},
                {"range": [40, 60], "color": "#78350F"},
                {"range": [60, 100],"color": "#14532D"},
            ],
            "threshold": {
                "line": {"color": "#1E2761", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Gauge shows the model-estimated probability that the selected stock outperforms SPY over the "
        "30 trading days following the hypothetical FOMC meeting. ≥60% is a strong positive signal; "
        "≤40% suggests likely underperformance; the band in between is uncertain. The model weighs "
        "pre-meeting momentum and volatility most heavily."
    )

    st.warning(
        "⚠️ **Exploratory model.** Trained on a small event-level sample (~30 FOMC events × 6 stocks) "
        "with chronological out-of-sample validation. Results may be unstable across market regimes and "
        "should not be used as standalone investment advice."
    )