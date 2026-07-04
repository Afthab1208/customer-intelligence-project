# Customer Intelligence Platform

End-to-end Data Science project on the real UCI Online Retail dataset
(541,909 transactions, Dec 2010 – Dec 2011, UK e-commerce company).

## What this project does
| Module | What it answers |
|---|---|
| cleaning.py | Handles missing IDs, cancellations, bad prices, duplicates |
| rfm_segmentation.py | WHO are our most valuable customers? |
| market_basket.py | WHICH products are bought together? |
| churn_model.py | WHO is about to stop buying from us? |
| app.py | Interactive dashboard for non-technical users |

## How to run
```
pip install -r requirements.txt
python src/data_loader.py       # confirm 541,909 rows load OK
python src/cleaning.py          # see cleaning summary
python src/rfm_segmentation.py  # see segments, saves data/rfm_segments.csv
python src/market_basket.py     # see top product rules, saves data/market_basket_rules.csv
python src/churn_model.py       # see model accuracy, saves data/churn_features.csv
streamlit run app.py            # launch interactive dashboard in browser
```

## Tech stack & why
| Tool | Purpose | Why not alternative |
|---|---|---|
| pandas/numpy | Data cleaning & aggregation | PySpark = overkill for <1M rows |
| scikit-learn KMeans | Customer segmentation | Simple, interpretable, scales to RFM |
| mlxtend Apriori | Market basket analysis | Standard implementation, no need to reinvent |
| Logistic Regression | Churn (interpretable) | Shows which features drive churn |
| Random Forest | Churn (accurate) | Handles non-linearity, gives feature importance |
| Streamlit | Dashboard | No HTML/CSS needed, built for data apps |
| Temporal train/test split | Prevent data leakage | Random split lets future data leak into past |
