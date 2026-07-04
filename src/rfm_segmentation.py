"""
rfm_segmentation.py  --  RFM scoring + KMeans customer segmentation.

RFM = Recency, Frequency, Monetary.  Industry-standard customer value framework.
KMeans groups customers by similarity on those three axes.
StandardScaler used because R/F/M are on different scales (days vs counts vs £).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency  = ("InvoiceDate",  lambda x: (snapshot - x.max()).days),
        Frequency= ("InvoiceNo",    "nunique"),
        Monetary = ("TotalPrice",   "sum"),
    ).reset_index()
    return rfm


def _quartile_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    """
    Rank-then-qcut avoids the common pd.qcut crash when many customers
    share the same value (e.g. lots of first-time buyers with Frequency=1).
    """
    scores = pd.qcut(series.rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
    return (5 - scores) if reverse else scores


def add_rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["R_Score"]   = _quartile_score(rfm["Recency"],   reverse=True)
    rfm["F_Score"]   = _quartile_score(rfm["Frequency"], reverse=False)
    rfm["M_Score"]   = _quartile_score(rfm["Monetary"],  reverse=False)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    return rfm


def _label_clusters(rfm: pd.DataFrame) -> pd.Series:
    """Label clusters by composite value score so labels are data-driven."""
    stats = rfm.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    z = (stats - stats.mean()) / stats.std(ddof=0)
    composite = z["Frequency"] + z["Monetary"] - z["Recency"]
    ranked = composite.sort_values(ascending=False).index.tolist()
    labels = ["Champions","Loyal Customers","Potential Loyalists","At Risk","Hibernating"]
    while len(labels) < len(ranked):
        labels.append(f"Segment {len(labels)+1}")
    mapping = {cluster: labels[i] for i, cluster in enumerate(ranked)}
    return rfm["Cluster"].map(mapping)


def cluster_customers(rfm: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    rfm = rfm.copy()
    scaled = StandardScaler().fit_transform(rfm[["Recency","Frequency","Monetary"]])
    rfm["Cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)
    rfm["Segment"] = _label_clusters(rfm)
    return rfm


def run_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    rfm = compute_rfm(df)
    rfm = add_rfm_scores(rfm)
    rfm = cluster_customers(rfm, n_clusters)
    print("\nSegment sizes:\n", rfm["Segment"].value_counts())
    print("\nSegment profiles:\n",
          rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1))
    return rfm


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    from data_loader import load_online_retail
    from cleaning import clean_online_retail
    rfm = run_segmentation(clean_online_retail(load_online_retail()))
    rfm.to_csv("data/rfm_segments.csv", index=False)
    print("\nSaved → data/rfm_segments.csv")
