
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
DB_DIR = ROOT / "data"
DB_PATH = DB_DIR / "instamart.db"

def main():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "customers": CLEAN_DIR / "customers.csv",
        "products": CLEAN_DIR / "products.csv",
        "orders": CLEAN_DIR / "orders.csv",
        "tickets": CLEAN_DIR / "tickets.csv",
    }

    with sqlite3.connect(DB_PATH) as connection:
        for table_name, csv_path in tables.items():
            df = pd.read_csv(csv_path)
            df.to_sql(
                table_name,
                connection,
                if_exists="replace",
                index=False
            )
            print(f"Loaded table '{table_name}': {len(df)} rows")

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_customer "
            "ON orders(customer_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_product "
            "ON orders(product_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_tickets_customer "
            "ON tickets(customer_id)"
        )

    print(f"\nDatabase created successfully: {DB_PATH}")

if __name__ == "__main__":
    main()
