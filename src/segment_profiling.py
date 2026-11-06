import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def profile_segments():
    rfm = pd.read_csv("outputs/part_b/rfm.csv")
    X = rfm[['recency', 'frequency', 'monetary']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Choose K=4 as optimal based on elbow and business interpretability
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm['segment_id'] = kmeans.fit_predict(X_scaled)
    
    # Profile segments
    profile = rfm.groupby('segment_id')[['recency', 'frequency', 'monetary']].mean()
    profile['size'] = rfm.groupby('segment_id').size()
    print("Segment Profiles (K=4):")
    print(profile)
    
    # Identify Premium segment (highest mean monetary)
    max_monetary_seg = profile['monetary'].idxmax()
    rfm['is_premium'] = (rfm['segment_id'] == max_monetary_seg)
    
    # Assign business names
    # Let's map segment IDs based on characteristics
    seg_names = {}
    for seg in profile.index:
        if seg == max_monetary_seg:
            seg_names[seg] = "Champions/Premium"
        elif profile.loc[seg, 'frequency'] > profile['frequency'].mean():
            seg_names[seg] = "Loyal Regulars"
        elif profile.loc[seg, 'recency'] > profile['recency'].mean():
            seg_names[seg] = "At-Risk/Lapsing"
        else:
            seg_names[seg] = "New/Occasional"
            
    rfm['segment_name'] = rfm['segment_id'].map(seg_names)
    
    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    
    # Save output
    rfm.to_csv("outputs/customer_segments.csv", index=False)
    print("outputs/customer_segments.csv saved successfully.")

if __name__ == "__main__":
    profile_segments()