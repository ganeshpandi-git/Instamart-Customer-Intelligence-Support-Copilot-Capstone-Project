
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ORDERS_PATH = ROOT / "data" / "clean" / "orders.csv"
OUTPUT_DIR = ROOT / "outputs" / "part_b"
OUTPUT_PATH = OUTPUT_DIR / "rfm.csv"

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orders = pd.read_csv(ORDERS_PATH)
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    completed = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] < SNAPSHOT)
    ].copy()

    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (
        SNAPSHOT - rfm["last_order_ts"]
    ).dt.days

    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ].sort_values("customer_id")

    rfm.to_csv(OUTPUT_PATH, index=False)

    print(f"RFM customers: {len(rfm)}")
    print("\nRFM summary:")
    print(rfm[["recency", "frequency", "monetary"]].describe())
    print(f"\nSaved: {OUTPUT_PATH}")

    assert len(rfm) == 1175, (
        f"Expected 1175 customers, found {len(rfm)}"
    )
    assert rfm["customer_id"].is_unique
    assert (rfm["frequency"] > 0).all()
    assert (rfm["monetary"] > 0).all()

    print("\nRFM validation passed.")

if __name__ == "__main__":
    main()
