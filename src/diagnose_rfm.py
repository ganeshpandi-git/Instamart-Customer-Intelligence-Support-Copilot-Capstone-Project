
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "data" / "clean" / "orders.csv"
snapshot = pd.Timestamp("2025-10-01 00:00:00")

orders = pd.read_csv(path)
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

print("Total cleaned order rows:", len(orders))
print("\nOrder status counts:")
print(orders["status"].value_counts(dropna=False).to_string())

completed = orders[orders["status"] == "completed"].copy()
eligible = completed[completed["order_ts"] < snapshot]

print("\nCompleted orders:", len(completed))
print("Completed orders before snapshot:", len(eligible))
print("Unique customers in eligible orders:",
      eligible["customer_id"].nunique())
print("Missing order timestamps:", orders["order_ts"].isna().sum())
print("Earliest order timestamp:", orders["order_ts"].min())
print("Latest order timestamp:", orders["order_ts"].max())
print("Completed orders on/after snapshot:",
      ((completed["order_ts"] >= snapshot)).sum())
