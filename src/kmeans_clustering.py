import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster_rfm():
    os.makedirs("outputs/part_b", exist_ok=True)
    rfm = pd.read_csv("outputs/part_b/rfm.csv")
    
    X = rfm[['recency', 'frequency', 'monetary']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_
        sil = silhouette_score(X_scaled, labels)
        results.append({"K": k, "inertia": inertia, "silhouette": sil})
        print(f"K={k}: Inertia={inertia:.1f}, Silhouette={sil:.3f}")
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/part_b/kmeans_metrics.csv", index=False)
    print("K-means metrics saved successfully.")

if __name__ == "__main__":
    cluster_rfm()