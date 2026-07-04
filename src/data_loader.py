"""
data_loader.py  --  loads the real UCI Online Retail dataset (541,909 rows).

KEY FIX vs common tutorials:
  ucimlrepo stores InvoiceNo & StockCode in .data.ids (not .data.features).
  We merge them back so all 8 columns are present.
"""

import os
import pandas as pd


def load_online_retail(local_csv_path: str = "data/online_retail.csv") -> pd.DataFrame:
    # --- Try ucimlrepo first (downloads automatically, no manual steps) ---
    try:
        from ucimlrepo import fetch_ucirepo
        print("Fetching dataset from UCI ML Repository...")
        repo = fetch_ucirepo(id=352)

        df = repo.data.features.copy().reset_index(drop=True)

        # ucimlrepo puts InvoiceNo & StockCode into .data.ids -- merge back
        if repo.data.ids is not None:
            ids = repo.data.ids.reset_index(drop=True)
            missing = [c for c in ids.columns if c not in df.columns]
            if missing:
                df = pd.concat([ids[missing], df], axis=1)

        print(f"Loaded {len(df):,} rows.  Columns: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"ucimlrepo failed ({e}). Trying local CSV...")

    # --- Fallback: local CSV (Kaggle manual download) ---
    if os.path.exists(local_csv_path):
        df = pd.read_csv(local_csv_path, encoding="ISO-8859-1")
        print(f"Loaded {len(df):,} rows from {local_csv_path}")
        return df

    raise FileNotFoundError(
        "Could not load data.  Either install ucimlrepo and have internet "
        "access, OR download the dataset manually from Kaggle "
        "('Online Retail Dataset') and place the CSV at 'data/online_retail.csv'."
    )


if __name__ == "__main__":
    df = load_online_retail()
    print(df.head())
    print(df.info())
