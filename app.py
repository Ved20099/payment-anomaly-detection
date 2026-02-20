import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from model import load_and_train

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sunrise Budget Inn — Payment Anomaly Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00326F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .anomaly-badge {
        background: #CC0000;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("Sunrise Budget Inn")
    st.caption("AI Payment Anomaly Detection")
    st.markdown("---")
    st.markdown("**Built by:** Ved Shastri")
    st.markdown("**Role:** Business & Financial Data Analyst")
    st.markdown("**Domain:** Banking · Payments · AI Systems")
    st.markdown("---")
    st.markdown("**Model:** Isolation Forest")
    st.markdown("**Dataset:** 8,311 transactions")
    st.markdown("**Period:** Jan–Dec 2024")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Executive Dashboard",
        "🚨 Flagged Transactions",
        "📈 Trend Analysis",
        "🔍 Live Transaction Check",
    ])

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_train("data/hotel_transactions.csv")

with st.spinner("Loading data and running anomaly detection model..."):
    df, model, scaler, features = get_data()

anomalies = df[df["predicted_anomaly"] == 1].copy()
normal    = df[df["predicted_anomaly"] == 0].copy()
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# ══════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":

    st.title("Payment Anomaly Detection System")
    st.markdown("*AI-powered transaction monitoring for Sunrise Budget Inn — Full Year 2024*")
    st.markdown("---")

    # ── KPI Row ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Transactions", f"{len(df):,}")
    with c2:
        st.metric("Total Revenue", f"${df[df['amount']>0]['amount'].sum():,.0f}")
    with c3:
        st.metric("🚨 Anomalies Flagged", f"{len(anomalies):,}",
                  delta=f"{len(anomalies)/len(df):.1%} of all transactions",
                  delta_color="inverse")
    with c4:
        at_risk = anomalies[anomalies["amount"] > 0]["amount"].sum()
        st.metric("💰 Revenue at Risk", f"${at_risk:,.0f}")
    with c5:
        avg_risk = anomalies["risk_score"].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.0f}/100")

    st.markdown("---")

    # ── Charts row 1 ─────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Anomaly Types Detected")
        type_counts = anomalies["anomaly_type"].value_counts().reset_index()
        type_counts.columns = ["Anomaly Type", "Count"]
        fig_bar = px.bar(
            type_counts, x="Count", y="Anomaly Type",
            orientation="h",
            color="Count",
            color_continuous_scale=["#1F5AA0", "#CC0000"],
            title="What the AI is flagging"
        )
        fig_bar.update_layout(showlegend=False, height=320,
                               plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("Transaction Amount: Normal vs Anomalous")
        fig_hist = px.histogram(
            df, x="amount",
            color=df["predicted_anomaly"].map({0: "Normal", 1: "Anomaly"}),
            nbins=60, opacity=0.75, barmode="overlay",
            color_discrete_map={"Normal": "#1F5AA0", "Anomaly": "#CC0000"},
            title="Anomalous transactions cluster at extreme amounts"
        )
        fig_hist.update_layout(height=320, plot_bgcolor="white",
                                paper_bgcolor="white", legend_title="")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("Anomalies by Payment Method")
        pm = anomalies["payment_method"].value_counts().reset_index()
        pm.columns = ["Payment Method", "Count"]
        fig_pie = px.pie(pm, values="Count", names="Payment Method",
                         color_discrete_sequence=["#00326F", "#1F5AA0", "#CC0000"],
                         title="Which payment channels are highest risk?")
        fig_pie.update_layout(height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r2:
        st.subheader("Anomalies by Hour of Day")
        hour_counts = anomalies["transaction_hour"].value_counts().sort_index().reset_index()
        hour_counts.columns = ["Hour", "Anomalies"]
        fig_hour = px.bar(
            hour_counts, x="Hour", y="Anomalies",
            color="Anomalies",
            color_continuous_scale=["#1F5AA0", "#CC0000"],
            title="1am–4am spike is a key anomaly signal"
        )
        fig_hour.update_layout(height=320, showlegend=False,
                                 plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_hour, use_container_width=True)

    # ── Monthly trend ────────────────────────────────────────
    st.subheader("Monthly Anomaly Trend — 2024")
    monthly = df.groupby(df["transaction_date"].dt.to_period("M")).agg(
        total=("transaction_id", "count"),
        anomalies=("predicted_anomaly", "sum"),
        revenue=("amount", lambda x: x[x > 0].sum())
    ).reset_index()
    monthly["transaction_date"] = monthly["transaction_date"].astype(str)
    monthly["anomaly_rate"] = (monthly["anomalies"] / monthly["total"] * 100).round(1)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=monthly["transaction_date"], y=monthly["total"],
        name="Total Transactions", marker_color="#1F5AA0", opacity=0.6
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly["transaction_date"], y=monthly["anomalies"],
        name="Anomalies", line=dict(color="#CC0000", width=3),
        mode="lines+markers", yaxis="y2"
    ))
    fig_trend.update_layout(
        yaxis=dict(title="Total Transactions"),
        yaxis2=dict(title="Anomalies", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        plot_bgcolor="white", paper_bgcolor="white", height=350
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — FLAGGED TRANSACTIONS
# ══════════════════════════════════════════════════════════════
elif page == "🚨 Flagged Transactions":

    st.title("🚨 Flagged Transactions — Analyst Review")
    st.markdown(f"**{len(anomalies):,} transactions** flagged by the AI model for review.")
    st.markdown("---")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        type_filter = st.multiselect(
            "Filter by Anomaly Type",
            options=anomalies["anomaly_type"].unique().tolist(),
            default=anomalies["anomaly_type"].unique().tolist()
        )
    with col_f2:
        pm_filter = st.multiselect(
            "Filter by Payment Method",
            options=anomalies["payment_method"].unique().tolist(),
            default=anomalies["payment_method"].unique().tolist()
        )
    with col_f3:
        min_risk = st.slider("Minimum Risk Score", 0, 100, 0)

    filtered = anomalies[
        (anomalies["anomaly_type"].isin(type_filter)) &
        (anomalies["payment_method"].isin(pm_filter)) &
        (anomalies["risk_score"] >= min_risk)
    ].sort_values("risk_score", ascending=False)

    st.markdown(f"Showing **{len(filtered)}** flagged transactions")

    display_cols = ["transaction_id", "booking_id", "transaction_date",
                    "transaction_time", "room_number", "room_type",
                    "amount", "payment_method", "transaction_type",
                    "anomaly_type", "risk_score", "staff_id"]

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=100, format="%d"
            ),
            "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
        }
    )

    # Download button
    csv = filtered[display_cols].to_csv(index=False)
    st.download_button(
        "📥 Download Flagged Transactions CSV",
        data=csv,
        file_name="flagged_transactions.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════════════════════
# PAGE 3 — TREND ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Trend Analysis":

    st.title("📈 Transaction Trend Analysis")
    st.markdown("---")

    # Revenue by room type
    st.subheader("Revenue by Room Type")
    rev_room = df[df["amount"] > 0].groupby("room_type")["amount"].sum().reset_index()
    rev_room.columns = ["Room Type", "Revenue"]
    fig_rev = px.bar(rev_room, x="Room Type", y="Revenue",
                     color="Revenue",
                     color_continuous_scale=["#1F5AA0", "#00326F"],
                     title="Total revenue contribution by room type — 2024")
    fig_rev.update_layout(showlegend=False, plot_bgcolor="white",
                           paper_bgcolor="white")
    st.plotly_chart(fig_rev, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Anomaly scatter
        st.subheader("Risk Score vs Transaction Amount")
        fig_sc = px.scatter(
            df.sample(min(3000, len(df))),
            x="amount", y="risk_score",
            color=df.sample(min(3000, len(df)))["predicted_anomaly"].map(
                {0: "Normal", 1: "Anomaly"}),
            color_discrete_map={"Normal": "#1F5AA0", "Anomaly": "#CC0000"},
            opacity=0.6,
            title="High-risk transactions cluster at extreme amounts"
        )
        fig_sc.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              legend_title="")
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        # Channel breakdown
        st.subheader("Anomalies by Booking Channel")
        ch = anomalies["channel"].value_counts().reset_index()
        ch.columns = ["Channel", "Anomalies"]
        fig_ch = px.bar(ch, x="Channel", y="Anomalies",
                        color="Anomalies",
                        color_continuous_scale=["#1F5AA0", "#CC0000"],
                        title="Which booking channels produce most anomalies?")
        fig_ch.update_layout(showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white")
        st.plotly_chart(fig_ch, use_container_width=True)

    # Staff analysis
    st.subheader("Anomalies by Staff ID — Internal Risk Monitor")
    staff_a = anomalies["staff_id"].value_counts().reset_index()
    staff_a.columns = ["Staff ID", "Anomalies Linked"]
    staff_t = df["staff_id"].value_counts().reset_index()
    staff_t.columns = ["Staff ID", "Total Transactions"]
    staff_merged = staff_a.merge(staff_t, on="Staff ID")
    staff_merged["Anomaly Rate"] = (
        staff_merged["Anomalies Linked"] / staff_merged["Total Transactions"] * 100
    ).round(1)
    st.dataframe(staff_merged, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — LIVE TRANSACTION CHECK
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Live Transaction Check":

    st.title("🔍 Live Transaction Anomaly Check")
    st.markdown("Enter any transaction details below and the AI will instantly assess its risk.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        amount       = st.number_input("Transaction Amount ($)", min_value=-1000.0,
                                        max_value=5000.0, value=95.0, step=5.0)
        room_type    = st.selectbox("Room Type",
                                    ["Standard Single", "Standard Double",
                                     "Queen Room", "Suite"])
        nights       = st.number_input("Number of Nights", min_value=1,
                                        max_value=30, value=1)
    with col2:
        pay_method   = st.selectbox("Payment Method",
                                    ["Credit Card", "Cash", "OTA (Booking.com)"])
        txn_hour     = st.slider("Transaction Hour (0–23)", 0, 23, 14)
        txn_type     = st.selectbox("Transaction Type",
                                    ["Room Charge", "Refund", "Parking Fee",
                                     "Pet Fee", "No-show Fee", "Late Check-out Fee"])
    with col3:
        base_rates   = {"Standard Single": 65, "Standard Double": 85,
                        "Queen Room": 95, "Suite": 140}
        base_rate    = base_rates[room_type]
        st.metric("Expected Base Rate", f"${base_rate}/night")
        st.metric("Expected Total", f"${base_rate * nights:.2f}")
        st.metric("Your Amount", f"${amount:.2f}")

    st.markdown("---")

    if st.button("🔍 Analyse This Transaction", type="primary", use_container_width=True):

        abs_amount        = abs(amount)
        amount_per_night  = abs_amount / max(nights, 1)
        is_refund         = 1 if amount < 0 else 0
        is_cash           = 1 if pay_method == "Cash" else 0
        is_odd_hour       = 1 if txn_hour <= 4 else 0
        rate_dev          = abs(amount_per_night - base_rate)
        rate_dev_pct      = rate_dev / base_rate if base_rate > 0 else 0

        import sklearn.preprocessing as pp
        test_row = pd.DataFrame([[
            abs_amount, amount_per_night, txn_hour, nights,
            base_rate, rate_dev, rate_dev_pct, is_cash, is_refund, is_odd_hour
        ]], columns=features)

        test_scaled  = scaler.transform(test_row)
        prediction   = model.predict(test_scaled)[0]
        score_raw    = model.decision_function(test_scaled)[0]

        min_s = df["anomaly_score"].min()
        max_s = df["anomaly_score"].max()
        risk  = round((score_raw - max_s) / (min_s - max_s) * 100, 1)
        risk  = max(0, min(100, risk))

        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            if prediction == -1:
                st.error(f"### ⚠️ ANOMALY DETECTED")
                st.metric("Risk Score", f"{risk}/100")
                st.markdown("**This transaction has been flagged for analyst review.**")
            else:
                st.success(f"### ✅ NORMAL TRANSACTION")
                st.metric("Risk Score", f"{risk}/100")
                st.markdown("**This transaction appears within normal parameters.**")

        with col_res2:
            st.markdown("**Why this score? Key signals checked:**")
            signals = {
                "Amount vs expected rate": f"${amount:.2f} vs ${base_rate * nights:.2f} expected — deviation {rate_dev_pct:.0%}",
                "Transaction hour":        f"{txn_hour}:00 {'⚠️ Unusual (1–4am)' if is_odd_hour else '✅ Normal hours'}",
                "Payment method":          f"{pay_method} {'⚠️ Large cash flagged' if is_cash and abs_amount > 300 else '✅ Normal'}",
                "Transaction type":        f"{txn_type} {'⚠️ Refund — review required' if is_refund else '✅ Normal'}",
            }
            for signal, detail in signals.items():
                st.markdown(f"- **{signal}:** {detail}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built by <b>Ved Shastri</b> · Business & Financial Data Analyst · "
    "Banking · Payments · AI Systems · "
    "Model: Isolation Forest | Python · Scikit-learn · Streamlit · Plotly</small></center>",
    unsafe_allow_html=True
)
