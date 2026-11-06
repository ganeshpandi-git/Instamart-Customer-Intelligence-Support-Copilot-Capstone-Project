
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs" / "part_b"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    query = """
    SELECT
        c.customer_id,
        c.city,
        c.age,
        c.gender,
        c.signup_date,
        COUNT(o.order_id) AS total_orders,
        SUM(
            CASE WHEN o.status = 'completed'
                 THEN o.order_amount ELSE 0 END
        ) AS completed_spend,
        SUM(
            CASE WHEN o.status = 'completed'
                 THEN 1 ELSE 0 END
        ) AS completed_orders,
        MAX(
            CASE WHEN o.status = 'completed'
                 THEN o.order_ts ELSE NULL END
        ) AS last_completed_order_ts,
        AVG(
            CASE WHEN o.status = 'completed'
                 THEN o.delivery_minutes ELSE NULL END
        ) AS avg_delivery_minutes
    FROM customers c
    LEFT JOIN orders o
        ON c.customer_id = o.customer_id
    GROUP BY
        c.customer_id, c.city, c.age, c.gender, c.signup_date
    ORDER BY c.customer_id;
    """

    with sqlite3.connect(DB_PATH) as connection:
        features = pd.read_sql_query(query, connection)

    features["signup_date"] = pd.to_datetime(
        features["signup_date"], errors="coerce"
    )
    features["last_completed_order_ts"] = pd.to_datetime(
        features["last_completed_order_ts"], errors="coerce"
    )

    features["completed_spend"] = (
        features["completed_spend"].fillna(0)
    )
    features["completed_orders"] = (
        features["completed_orders"].fillna(0).astype(int)
    )
    features["total_orders"] = (
        features["total_orders"].fillna(0).astype(int)
    )

    output_path = OUTPUT_DIR / "customer_features.csv"
    features.to_csv(output_path, index=False)

    print(f"Customers in feature dataset: {len(features)}")
    print(f"Columns: {len(features.columns)}")
    print(f"Saved: {output_path}")
    print("\nPreview:")
    print(features.head(5).to_string(index=False))
    print("\nMissing values by column:")
    print(features.isna().sum().to_string())

if __name__ == "__main__":
    main()
