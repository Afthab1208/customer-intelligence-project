"""
churn_model.py  --  predicts which customers are likely to churn.

CRITICAL interview point -- WHY a temporal split (not random):
  A random split lets "future" data leak into "past" predictions for the
  same customer, making accuracy look artificially perfect.  Splitting by
  date means: features come ONLY from before the cutoff, labels come ONLY
  from after.  This mirrors real production use -- predicting forward.

WHY Logistic Regression + Random Forest (not deep learning):
  Dataset is small + tabular.  Tree-based/linear models beat neural nets
  here.  LR gives interpretable coefficients; RF gives feature importances.
  Both are easy to explain to a non-technical interviewer.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_churn_dataset(df: pd.DataFrame, observation_days: int = 90) -> pd.DataFrame:
    max_date    = df["InvoiceDate"].max()
    cutoff_date = max_date - pd.Timedelta(days=observation_days)
    history     = df[df["InvoiceDate"] <  cutoff_date]
    future      = df[df["InvoiceDate"] >= cutoff_date]

    if history.empty:
        raise ValueError("observation_days too large -- no history before cutoff.")

    features = history.groupby("CustomerID").agg(
        Recency       = ("InvoiceDate", lambda x: (cutoff_date - x.max()).days),
        Frequency     = ("InvoiceNo",   "nunique"),
        Monetary      = ("TotalPrice",  "sum"),
        AvgOrderValue = ("TotalPrice",  lambda x: x.sum() / max(history.loc[x.index, "InvoiceNo"].nunique(), 1)),
        UniqueProducts= ("Description", "nunique"),
        TenureDays    = ("InvoiceDate", lambda x: (cutoff_date - x.min()).days),
    ).reset_index()

    active_after = set(future["CustomerID"].unique())
    features["Churned"] = features["CustomerID"].apply(
        lambda c: 0 if c in active_after else 1   # 1 = churned
    )
    return features


def train_churn_models(features: pd.DataFrame) -> dict:
    FEAT = ["Recency","Frequency","Monetary","AvgOrderValue","UniqueProducts","TenureDays"]
    X, y = features[FEAT], features["Churned"]

    if y.nunique() < 2:
        print("Only one churn class present -- increase observation_days or use more data.")
        return {"logistic_regression": None, "random_forest": None,
                "scaler": StandardScaler().fit(X), "feature_cols": FEAT}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler        = StandardScaler()
    X_train_sc    = scaler.fit_transform(X_train)
    X_test_sc     = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    print("=== Logistic Regression ===")
    print(classification_report(y_test, lr.predict(X_test_sc), zero_division=0))
    if len(set(y_test)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:,1]):.3f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                                class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    print("\n=== Random Forest ===")
    print(classification_report(y_test, rf.predict(X_test), zero_division=0))
    if len(set(y_test)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.3f}")
    print("\nFeature importances:")
    for f, i in sorted(zip(FEAT, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f}: {i:.3f}")

    return {"logistic_regression": lr, "random_forest": rf,
            "scaler": scaler, "feature_cols": FEAT}


def run_churn_pipeline(df: pd.DataFrame, observation_days: int = 90):
    features = build_churn_dataset(df, observation_days)
    print(f"Churn dataset: {len(features)} customers, "
          f"{features['Churned'].mean():.1%} churned.\n")
    models = train_churn_models(features)
    return features, models


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    from data_loader import load_online_retail
    from cleaning import clean_online_retail
    df = clean_online_retail(load_online_retail())
    features, models = run_churn_pipeline(df)
    features.to_csv("data/churn_features.csv", index=False)
    print("\nSaved → data/churn_features.csv")
