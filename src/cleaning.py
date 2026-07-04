"""
cleaning.py  --  cleans raw Online Retail transactions.

Every decision is documented with WHY, so you can defend it in an interview.
"""

import pandas as pd


def clean_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Steps (each justified):
    1. Drop missing CustomerID    -- no customer = can't do any customer analysis
    2. Remove cancellations (C*)  -- InvoiceNo starting with C = reversal, not a sale
    3. Remove qty <= 0            -- returns / data errors, not real purchases
    4. Remove price <= 0          -- samples / adjustments, not real revenue
    5. Convert InvoiceDate        -- need datetime for recency & time analysis
    6. Drop exact duplicates      -- double-scanned entries inflate counts
    7. Add TotalPrice column      -- Quantity * UnitPrice = line-item revenue
    8. Tidy text fields           -- trim whitespace so IDs match consistently
    """
    data = df.copy()
    before = len(data)

    # 1. missing CustomerID
    data = data.dropna(subset=["CustomerID"])

    # 2. cancellations
    data["InvoiceNo"] = data["InvoiceNo"].astype(str).str.strip()
    data = data[~data["InvoiceNo"].str.startswith("C")]

    # 3 & 4. non-positive qty / price
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]

    # 5. datetime
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

    # 6. duplicates
    data = data.drop_duplicates()

    # 7. revenue column
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    # 8. tidy text
    data["CustomerID"] = data["CustomerID"].astype(int).astype(str).str.strip()
    data["Country"]     = data["Country"].astype(str).str.strip()
    data["Description"] = data["Description"].astype(str).str.strip()

    after = len(data)
    print(f"Cleaning: {before:,} → {after:,} rows  ({100*(before-after)/before:.1f}% removed)")
    return data.reset_index(drop=True)


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    from data_loader import load_online_retail
    df = clean_online_retail(load_online_retail())
    print(df.head())
    print(df.describe())
