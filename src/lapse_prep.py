import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_lapse_data():
    os.makedirs("outputs/part_b", exist_ok=True)
    
    orders = pd.read_csv("data/clean/orders.csv")
    customers = pd.read_csv("data/clean/customers.csv")
    products = pd.read_csv("data/clean/products.csv")
    
    orders['order_ts'] = pd.to_datetime(orders['order_ts'])
    customers['signup_date'] = pd.to_datetime(customers['signup_date'])
    
    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    
    # Cohort: customers with at least one completed order before cutoff
    pre_cutoff_completed = orders[(orders['status'] == 'completed') & (orders['order_ts'] < cutoff)]
    cohort_customers = pre_cutoff_completed['customer_id'].unique()
    
    print(f"Cohort customers before cutoff: {len(cohort_customers)}")
    
    # Label: lapsed = 1 if NO completed order between 2025-09-01 and 2025-09-30
    sept_completed = orders[(orders['status'] == 'completed') & 
                            (orders['order_ts'] >= cutoff) & 
                            (orders['order_ts'] <= pd.to_datetime("2025-09-30 23:59:59"))]
    active_in_sept = sept_completed['customer_id'].unique()
    
    labels = []
    for cid in cohort_customers:
        lapsed = 0 if cid in active_in_sept else 1
        labels.append({'customer_id': cid, 'lapsed': lapsed})
        
    labels_df = pd.DataFrame(labels)
    print(f"Lapsed count: {labels_df['lapsed'].sum()} ({labels_df['lapsed'].mean()*100:.2f}%)")
    
    # Build features using only orders before cutoff
    sub_orders = orders[orders['order_ts'] < cutoff]
    sub_orders = sub_orders.merge(products[['product_id', 'category']], on='product_id', how='left')
    
    # Recency from cutoff
    recency = sub_orders[sub_orders['status'] == 'completed'].groupby('customer_id')['order_ts'].max().apply(lambda x: (cutoff - x).days)
    recency.name = 'recency_to_cutoff'
    
    # Frequency & monetary
    comp_orders = sub_orders[sub_orders['status'] == 'completed']
    freq = comp_orders.groupby('customer_id')['order_id'].count().rename('frequency')
    mon = comp_orders.groupby('customer_id')['order_amount'].sum().rename('monetary')
    avg_disc = comp_orders.groupby('customer_id')['discount_pct'].mean().rename('avg_discount')
    avg_deliv = comp_orders.groupby('customer_id')['delivery_minutes'].mean().rename('avg_delivery')
    
    # Cancellation rate before cutoff
    all_status = sub_orders.groupby('customer_id')['status'].agg(
        total=lambda x: (x.isin(['completed', 'cancelled'])).sum(),
        cancelled=lambda x: (x == 'cancelled').sum()
    )
    cancel_rate = (all_status['cancelled'] / all_status['total'].replace(0, 1)).rename('cancellation_rate')
    
    # Most frequent product category
    cat_counts = sub_orders.groupby(['customer_id', 'category']).size().reset_index(name='count')
    fav_cat = cat_counts.sort_values(['customer_id', 'count'], ascending=[True, False]).drop_duplicates('customer_id').set_index('customer_id')['category'].rename('favourite_category')
    
    # Combine features with customer demographics
    feat_df = pd.DataFrame({'customer_id': cohort_customers}).set_index('customer_id')
    feat_df = feat_df.join([recency, freq, mon, avg_disc, avg_deliv, cancel_rate, fav_cat])
    
    cust_feat = customers.set_index('customer_id')[['city', 'age', 'gender', 'signup_date']]
    feat_df = feat_df.join(cust_feat)
    
    feat_df['tenure_days'] = (cutoff - feat_df['signup_date']).dt.days
    feat_df = feat_df.drop(columns=['signup_date']).reset_index()
    
    # Fill missing values for numerical features appropriately
    feat_df['recency_to_cutoff'] = feat_df['recency_to_cutoff'].fillna(183)
    feat_df['frequency'] = feat_df['frequency'].fillna(0)
    feat_df['monetary'] = feat_df['monetary'].fillna(0)
    feat_df['avg_discount'] = feat_df['avg_discount'].fillna(0)
    feat_df['avg_delivery'] = feat_df['avg_delivery'].fillna(feat_df['avg_delivery'].median())
    feat_df['cancellation_rate'] = feat_df['cancellation_rate'].fillna(0)
    feat_df['favourite_category'] = feat_df['favourite_category'].fillna('Grocery')
    
    final_df = feat_df.merge(labels_df, on='customer_id')
    final_df.to_csv("outputs/part_b/lapse_dataset.csv", index=False)
    print(f"Lapse dataset saved. Shape: {final_df.shape}")

if __name__ == "__main__":
    prepare_lapse_data()