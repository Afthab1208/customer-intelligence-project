"""
market_basket.py  --  finds products frequently bought together (Apriori).

Key metrics you must know for interviews:
  Support    = how often the pair appears across all invoices
  Confidence = P(buy B | bought A)
  Lift       = how much MORE likely B is bought with A vs by chance.
               Lift > 1 = genuine positive association.
               We filter on Lift (not just Confidence) because a very popular
               product will have high confidence with everything, which is
               misleading -- Lift corrects for that.
"""

import pandas as pd


def build_basket_matrix(df: pd.DataFrame, country: str = "United Kingdom") -> pd.DataFrame:
    data = df[df["Country"] == country] if country else df
    basket = (
        data.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum().unstack().fillna(0)
    )
    return (basket > 0).astype(int)


def run_market_basket(
    df: pd.DataFrame,
    country: str = "United Kingdom",
    min_support: float = 0.02,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    from mlxtend.frequent_patterns import apriori, association_rules

    basket = build_basket_matrix(df, country)
    frequent = apriori(basket, min_support=min_support, use_colnames=True)

    if frequent.empty:
        print("No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(frequent, metric="lift", min_threshold=min_lift)
    rules = rules.sort_values("lift", ascending=False)

    # Convert frozensets to readable strings
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    keep = [c for c in ["antecedents","consequents","support","confidence","lift"]
            if c in rules.columns]
    rules = rules[keep].reset_index(drop=True)
    print(f"Found {len(rules)} rules for {country}.")
    print(rules.head(10))
    return rules


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    from data_loader import load_online_retail
    from cleaning import clean_online_retail
    df = clean_online_retail(load_online_retail())
    rules = run_market_basket(df)
    rules.to_csv("data/market_basket_rules.csv", index=False)
    print("\nSaved → data/market_basket_rules.csv")
