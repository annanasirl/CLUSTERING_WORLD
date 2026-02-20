import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

HIER_FOLDER = "../data/SERIES/SERIES_hierarchical_2_20260220_175147"
DBSCAN_FOLDER = "../data/SERIES/SERIES_dbscan_2_20260220_175245"
COUNTRY_COL = "Country Code"
CLUSTER_COL = "cluster"


# --- Funzione per calcolare centroidi ---
def compute_centroids(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
    centroids_by_year = {}
    for f in files:
        year = int("".join(filter(str.isdigit, f)))
        df = pd.read_csv(os.path.join(folder, f))
        features = df.drop(columns=[COUNTRY_COL, CLUSTER_COL])
        features = features.select_dtypes(include=[np.number])
        centroids = {}
        for cluster in df[CLUSTER_COL].unique():
            if cluster == -1:
                continue
            cluster_data = features[df[CLUSTER_COL] == cluster]
            centroids[cluster] = cluster_data.mean()
        centroids_by_year[year] = centroids
    return centroids_by_year


# --- Funzione per stampare summary ---
def print_centroid_summary(centroids_dict, name="HIER"):
    print(f"\n=== {name} Centroid Summary ===")
    for year, clusters in sorted(centroids_dict.items()):
        print(f"\nYear: {year}")
        for cl, centroid in clusters.items():
            print(f" Cluster {cl}: {centroid.head(5).to_dict()} ...")


# --- Funzione per cosine shift ---
def compute_cosine_shifts(centroids_dict):
    shifts_by_cluster = {}
    years = sorted(centroids_dict.keys())
    for cluster in set().union(*[centroids_dict[y].keys() for y in years]):
        shifts = []
        prev_centroid = None
        for year in years:
            if cluster not in centroids_dict[year]:
                continue
            centroid = centroids_dict[year][cluster]
            if prev_centroid is not None:
                dist = cosine(prev_centroid.values, centroid.values)
                shifts.append((year, dist))
            prev_centroid = centroid
        if shifts:
            shifts_by_cluster[cluster] = shifts
    return shifts_by_cluster


# --- Funzione per trasformare centroidi in DataFrame ---
def centroids_to_dataframe(centroids_dict):
    rows = []
    for year, clusters in centroids_dict.items():
        for cl, centroid in clusters.items():
            row = centroid.copy()
            row['cluster'] = cl
            row['year'] = year
            rows.append(row)
    return pd.DataFrame(rows)


# --- Calcolo centroidi ---
hier_centroids = compute_centroids(HIER_FOLDER)
dbscan_centroids = compute_centroids(DBSCAN_FOLDER)

# --- Summary centroidi ---
print_centroid_summary(hier_centroids, "HIER")
print_centroid_summary(dbscan_centroids, "DBSCAN")

# --- Cosine shifts ---
hier_shifts = compute_cosine_shifts(hier_centroids)
dbscan_shifts = compute_cosine_shifts(dbscan_centroids)

print("\n=== Cosine Shifts HIER ===")
for cl, shifts in hier_shifts.items():
    print(f"HIER Cluster {cl}: {[f'Year {y}: {d:.3f}' for y, d in shifts]}")

print("\n=== Cosine Shifts DBSCAN ===")
for cl, shifts in dbscan_shifts.items():
    print(f"DBSCAN Cluster {cl}: {[f'Year {y}: {d:.3f}' for y, d in shifts]}")


# --- PCA e visualizzazione separata ---
def plot_centroids_pca(centroids_dict, title="Centroid movement (PCA 2D)"):
    df_centroids = centroids_to_dataframe(centroids_dict)
    feature_cols = df_centroids.columns.difference(['cluster', 'year'])
    scaler = StandardScaler()
    df_centroids[feature_cols] = scaler.fit_transform(df_centroids[feature_cols])

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(df_centroids[feature_cols])
    df_centroids['PC1'] = pca_coords[:, 0]
    df_centroids['PC2'] = pca_coords[:, 1]

    plt.figure(figsize=(10, 6))
    for cl in df_centroids['cluster'].unique():
        df_c = df_centroids[df_centroids['cluster'] == cl].sort_values('year')
        plt.plot(df_c['PC1'], df_c['PC2'], marker='o', label=f"Cluster {cl}")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Plot separati ---
plot_centroids_pca(hier_centroids, title="HIER Centroid movement over time (PCA 2D)")
plot_centroids_pca(dbscan_centroids, title="DBSCAN Centroid movement over time (PCA 2D)")

import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_shifts(shifts_dict, title="Cumulative Cosine Shift"):
    plt.figure(figsize=(10,6))
    for cl, shifts in shifts_dict.items():
        years, vals = zip(*shifts)
        cumulative = np.cumsum(vals)
        plt.plot(years, cumulative, marker='o', label=f"Cluster {cl}")
    plt.xlabel("Year")
    plt.ylabel("Cumulative Cosine Shift")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Grafici cumulativi separati ---
plot_cumulative_shifts(hier_shifts, title="HIER Cumulative Cosine Shift over Time")
plot_cumulative_shifts(dbscan_shifts, title="DBSCAN Cumulative Cosine Shift over Time")

import seaborn as sns

# --- Funzione per creare heatmap di un singolo cluster ---
def plot_cluster_heatmap(centroids_dict, cluster_id, cluster_name="Cluster", title_suffix=""):
    # Ricavo solo i centroidi del cluster richiesto
    cluster_rows = []
    for year, clusters in centroids_dict.items():
        if cluster_id in clusters:
            row = clusters[cluster_id].copy()
            row.name = year
            cluster_rows.append(row)
    if not cluster_rows:
        print(f"Nessun dato per {cluster_name} {cluster_id}")
        return

    df_cluster = pd.DataFrame(cluster_rows)
    df_cluster = df_cluster.T  # Trasposta: righe=attributi, colonne=anni

    plt.figure(figsize=(12, max(6, len(df_cluster)/2)))
    sns.heatmap(df_cluster, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Value'})
    plt.title(f"{cluster_name} {cluster_id} Centroid Attributes {title_suffix}")
    plt.xlabel("Year")
    plt.ylabel("Attribute")
    plt.tight_layout()
    plt.show()


# --- Heatmap desiderate ---
# DBSCAN cluster 1
plot_cluster_heatmap(dbscan_centroids, cluster_id=1, cluster_name="DBSCAN", title_suffix="over years")

# DBSCAN cluster 2
plot_cluster_heatmap(dbscan_centroids, cluster_id=2, cluster_name="DBSCAN", title_suffix="over years")

# HIER cluster 0
plot_cluster_heatmap(hier_centroids, cluster_id=0, cluster_name="HIER", title_suffix="over years")

# HIER cluster 1
plot_cluster_heatmap(hier_centroids, cluster_id=1, cluster_name="HIER", title_suffix="over years")

# --- Definisco i meta-cluster per HIER ---
meta_cluster_0_years = [2005, 2006, 2007, 2008, 2011, 2012, 2014, 2015, 2017]
meta_cluster_1_years = [2009, 2010, 2013, 2016] + list(range(2018, 2023))  # dal 2018 al 2022

# Funzione per ricombinare i cluster secondo le regole date
def recombine_hier_clusters(centroids_dict, cluster_0_years, cluster_1_years):
    new_centroids = {}
    for year, clusters in centroids_dict.items():
        new_centroids[year] = {}
        if year in cluster_0_years:
            if 0 in clusters:
                new_centroids[year][0] = clusters[0]  # cluster 0 → meta-cluster 0
            if 1 in clusters:
                new_centroids[year][1] = clusters[1]  # cluster 1 → meta-cluster 1
        elif year in cluster_1_years:
            if 1 in clusters:
                new_centroids[year][0] = clusters[1]  # cluster 1 → meta-cluster 0
            if 0 in clusters:
                new_centroids[year][1] = clusters[0]  # cluster 0 → meta-cluster 1
    return new_centroids

# --- Ricombino i cluster ---
hier_recombined = recombine_hier_clusters(hier_centroids, meta_cluster_0_years, meta_cluster_1_years)

# --- Plot heatmap dei meta-cluster ---
plot_cluster_heatmap(hier_recombined, cluster_id=0, cluster_name="HIER Meta-Cluster 0", title_suffix="over years")
plot_cluster_heatmap(hier_recombined, cluster_id=1, cluster_name="HIER Meta-Cluster 1", title_suffix="over years")