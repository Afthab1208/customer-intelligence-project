"""
app.py  --  Streamlit dashboard.
Run with:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from data_loader      import load_online_retail
from cleaning         import clean_online_retail
from rfm_segmentation import compute_rfm, add_rfm_scores, cluster_customers
from churn_model      import build_churn_dataset, train_churn_models

st.set_page_config(page_title="Customer Intelligence", layout="wide")
st.title("Customer Intelligence Dashboard")
st.caption("Online Retail dataset  |  Segmentation · Market Basket · Churn Prediction")


@st.cache_data
def get_clean_data():
    return clean_online_retail(load_online_retail())

@st.cache_data
def get_rfm(clean_df):
    rfm = compute_rfm(clean_df)
    rfm = add_rfm_scores(rfm)
    return cluster_customers(rfm, n_clusters=4)

@st.cache_data
def get_basket_rules(clean_df, country):
    try:
        from market_basket import run_market_basket
        return run_market_basket(clean_df, country=country), None
    except Exception as e:
        return None, str(e)

@st.cache_data
def get_churn(clean_df):
    features = build_churn_dataset(clean_df, observation_days=90)
    models   = train_churn_models(features)
    return features, models


with st.spinner("Loading & cleaning data (one-time, ~30 sec)..."):
    clean_df = get_clean_data()

# ── KPI strip ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue",    f"£{clean_df['TotalPrice'].sum():,.0f}")
k2.metric("Unique Customers", f"{clean_df['CustomerID'].nunique():,}")
k3.metric("Total Orders",     f"{clean_df['InvoiceNo'].nunique():,}")
k4.metric("Avg Order Value",
          f"£{clean_df.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.2f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Segments", "🛒 Market Basket", "⚠️ Churn Risk", "🔍 Customer Lookup"])

# ── Tab 1: Segments ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("RFM Customer Segments (KMeans, k=4)")
    rfm = get_rfm(clean_df)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("**Segment sizes**")
        st.bar_chart(rfm["Segment"].value_counts())
    with c2:
        st.write("**Mean Recency / Frequency / Monetary per segment**")
        st.dataframe(
            rfm.groupby("Segment")[["Recency","Frequency","Monetary"]]
            .mean().round(1)
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=rfm, x="Recency", y="Monetary",
                    hue="Segment", size="Frequency", sizes=(20, 200), ax=ax)
    ax.set_title("Recency vs Monetary (bubble = Frequency)")
    st.pyplot(fig)

    with st.expander("Full segment table"):
        st.dataframe(rfm.sort_values("Monetary", ascending=False))

# ── Tab 2: Market Basket ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Frequently Bought Together (Apriori)")
    country = st.selectbox("Filter by country",
                           clean_df["Country"].value_counts().index.tolist())
    rules, err = get_basket_rules(clean_df, country)
    if err:
        st.warning(f"Market basket error: {err}")
    elif rules is None or rules.empty:
        st.info("No strong rules found. Try selecting 'United Kingdom' (most data).")
    else:
        st.dataframe(rules.head(20))
        st.caption("Lift > 1 = these products are bought together more than chance predicts.")

# ── Tab 3: Churn ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Churn Prediction (90-day observation window)")
    with st.spinner("Training models..."):
        churn_feat, models = get_churn(clean_df)

    rf         = models["random_forest"]
    feat_cols  = models["feature_cols"]
    churn_feat = churn_feat.copy()
    churn_feat["ChurnProbability"] = rf.predict_proba(churn_feat[feat_cols])[:, 1]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("**Feature importance**")
        imp = pd.DataFrame({"feature": feat_cols,
                            "importance": rf.feature_importances_})\
                .sort_values("importance", ascending=False)
        st.bar_chart(imp.set_index("feature"))
    with c2:
        st.write("**Highest churn-risk customers**")
        st.dataframe(
            churn_feat.sort_values("ChurnProbability", ascending=False)
            .head(20)[["CustomerID","Recency","Frequency","Monetary","ChurnProbability"]]
        )

# ── Tab 4: Customer Lookup ────────────────────────────────────────────────────
with tab4:
    st.subheader("Look up a single customer")
    rfm = get_rfm(clean_df)
    cid = st.selectbox("Select CustomerID", rfm["CustomerID"].tolist())
    row = rfm[rfm["CustomerID"] == cid].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Segment",           row["Segment"])
    m2.metric("Recency (days)",    int(row["Recency"]))
    m3.metric("Orders",            int(row["Frequency"]))
    m4.metric("Total Spend (£)",   f"{row['Monetary']:,.2f}")

    st.write("**Transaction history**")
    st.dataframe(
        clean_df[clean_df["CustomerID"] == cid]
        .sort_values("InvoiceDate", ascending=False)
    )
