import pandas as pd
import glob
import os
import numpy as np

# --- Funzioni ---
def cosine_distance(a, b):
    """Calcola la distanza coseno tra due vettori 2D"""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a = a / np.maximum(a_norm, 1e-10)
    b = b / np.maximum(b_norm, 1e-10)
    return 1 - np.dot(a, b.T)

def silhouette_cosine(X, labels):
    """Calcolo silhouette score usando distanza coseno"""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n_samples = X.shape[0]

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan  # silhouette score non definita

    score = 0
    for i in range(n_samples):
        own_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == k] for k in unique_labels if k != labels[i]]

        # a(i) = distanza media dal proprio cluster
        if len(own_cluster) > 1:
            a_i = np.mean([cosine_distance(X[i:i+1], p.reshape(1, -1))[0, 0]
                           for p in own_cluster if not np.array_equal(p, X[i])])
        else:
            a_i = 0

        # b(i) = minima distanza media dagli altri cluster
        b_i = np.min([np.mean([cosine_distance(X[i:i+1], p.reshape(1, -1))[0, 0]
                               for p in cluster])
                      for cluster in other_clusters])

        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        score += s_i

    return score / n_samples

# --- Cartelle clustering finali ---
folders = {
    "HIER": "../data/SERIES/SERIES_hierarchical_2_20260220_175147",
    "DBSCAN": "../data/SERIES/SERIES_dbscan_2_20260220_175245",
    "KMEANS_4": "../data/SERIES/SERIES_k-means_4_20260220_175024",
    "CLARA_6": "../data/SERIES/SERIES_clara_6_20260220_175058"  # aggiungi se necessario
}

all_results = {}

for algo_name, folder in folders.items():
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    results = []
    for file in files:
        filename = os.path.basename(file)
        # estrai anno dai numeri nel filename
        year = int("".join(filter(str.isdigit, filename)))

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        features = df.drop(columns=["Country Code", "cluster"], errors='ignore')
        features = features.select_dtypes(include=['number']).values
        labels = df["cluster"].values

        score = silhouette_cosine(features, labels)
        results.append({"Year": year, algo_name: score})

    algo_df = pd.DataFrame(results).sort_values("Year").reset_index(drop=True)
    all_results[algo_name] = algo_df.set_index("Year")[algo_name]

# --- Unione risultati in un unico DataFrame ---
silhouette_all_df = pd.DataFrame(all_results)
silhouette_all_df.index.name = "Year"
silhouette_all_df = silhouette_all_df.sort_index()

# --- Salvataggio CSV ---
output_dir = "../data/series analysis/silhouette_cosine/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "silhouette_scores_all_algorithms.csv")
silhouette_all_df.to_csv(output_path)
print("Silhouette scores per algoritmo salvati in:", output_path)
print(silhouette_all_df)