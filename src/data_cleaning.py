import os
import sqlite3
import pandas as pd
import numpy as np

def clean_data():
    os.makedirs("data/clean", exist_ok=True)
    os.makedirs("outputs/part_a", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Load raw data
    cust_raw = pd.read_csv("data/raw/customers.csv", keep_default_na=False)
    prod_raw = pd.read_csv("data/raw/products.csv", keep_default_na=False)
    orders_raw = pd.read_csv("data/raw/orders.csv", keep_default_na=False)
    tickets_raw = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

    # Clean Customers
    customers = cust_raw.copy()
    customers['city'] = customers['city'].str.strip().str.title()
    
    def clean_gender(g):
        g_str = str(g).strip().lower()
        if g_str in ['male', 'm']: return 'male'
        elif g_str in ['female', 'f']: return 'female'
        else: return 'unknown'

    customers['gender'] = customers['gender'].apply(clean_gender)
    customers['age'] = pd.to_numeric(customers['age'], errors='coerce')
    customers.loc[(customers['age'] < 10) | (customers['age'] > 100), 'age'] = np.nan
    customers['age'] = customers['age'].fillna(customers.groupby('city')['age'].transform('median'))
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])

    # Clean Products
    products = prod_raw.copy()
    products['price_inr'] = pd.to_numeric(products['price_inr'])

    # Clean Orders following exact brief rules:
    orders = orders_raw.copy()
    orders['order_ts'] = pd.to_datetime(orders['order_ts'])

    # 1. Remove exact duplicate rows
    orders = orders.drop_duplicates()

    # 2. Keep only latest order_ts per order_id (handles pending + final records)
    orders = orders.sort_values('order_ts').drop_duplicates(subset=['order_id'], keep='last')

    # 3. Convert order_amount to numeric, fill missing/unparseable using formula
    orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
    orders = orders.merge(products[['product_id', 'price_inr', 'category']], on='product_id', how='left')
    
    calc_amt = round(orders['price_inr'] * orders['quantity'] * (1 - orders['discount_pct'] / 100), 2)
    orders['order_amount'] = orders['order_amount'].fillna(calc_amt)

    # 4. Handle delivery_minutes
    orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
    orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan

    # Impute completed order delivery minutes with median of category
    cat_medians = orders[orders['status'] == 'completed'].groupby('category')['delivery_minutes'].median()
    for cat, med in cat_medians.items():
        mask = (orders['status'] == 'completed') & (orders['category'] == cat) & (orders['delivery_minutes'].isna())
        orders.loc[mask, 'delivery_minutes'] = med

    # Cancelled orders must have missing delivery minutes
    orders.loc[orders['status'] == 'cancelled', 'delivery_minutes'] = np.nan

    orders = orders[['order_id', 'customer_id', 'product_id', 'quantity', 'order_ts', 'status', 'discount_pct', 'order_amount', 'delivery_minutes']]

    # Save clean tables
    customers.to_csv("data/clean/customers.csv", index=False)
    products.to_csv("data/clean/products.csv", index=False)
    orders.to_csv("data/clean/orders.csv", index=False)
    tickets_raw.to_csv("data/clean/tickets.csv", index=False)

    # Store in SQLite
    conn = sqlite3.connect("data/instamart.db")
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    
    orders_db = orders.copy()
    orders_db['order_ts'] = orders_db['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    orders_db.to_sql("orders", conn, if_exists="replace", index=False)
    tickets_raw.to_sql("tickets", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Clean orders rows: {len(orders)} (Completed: {(orders['status']=='completed').sum()}, Cancelled: {(orders['status']=='cancelled').sum()})")

if __name__ == "__main__":
    clean_data()