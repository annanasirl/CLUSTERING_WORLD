import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def hopkins_statistic(X, n_samples=None, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)

    n, d = X.shape
    if n_samples is None:
        n_samples = max(1, int(0.1 * n))
    # di base prendo il 10 per cento del data set

    # campiono i punti reali del dataset a random senza replacement
    idx = rng.choice(n, n_samples, replace=False)
    X_real = X[idx]

    # prendo punti random casuali tanti quanti quelli reali, uniforme
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_rand = rng.uniform(X_min, X_max, size=(n_samples, d))

    # runno nn x trovare neighb. di pt reali
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    # u sono le distanze pt reale - pt reale
    u_dist, _ = nbrs.kneighbors(X_real, n_neighbors=2)
    u = u_dist[:, 1]

    # w sono le distanze pt reale - pt random
    w_dist, _ = nbrs.kneighbors(X_rand, n_neighbors=1)
    w = w_dist[:, 0]

    # ritorno la hopkins statistic
    return np.sum(w) / (np.sum(w) + np.sum(u))

def run_hopkins_on_csv(
    csv_path,
    n_runs=30,
    random_state=42
):
    df = pd.read_csv(csv_path)

    # seleziono le feature che mi interssano
    feature_cols = df.columns.difference(["Country Name", "Country Code"])
    X = df[feature_cols]

    # dato che nel dataset ho i null value, x fare hopk fillo
    X = X.fillna(X.mean())

    # normalizzo
    X = StandardScaler().fit_transform(X)

    # Hopkins ripetuta per n runs
    values = []
    for i in range(n_runs):
        H = hopkins_statistic(X, random_state=random_state + i)
        values.append(H)

    print("Hopkins Statistic")
    print(f"Mean: {np.mean(values):.4f}")
    print(f"Std:  {np.std(values):.4f}")
    print(f"Min:  {np.min(values):.4f}")
    print(f"Max:  {np.max(values):.4f}")

    return values

def hopkins_on_pca(
    csv_path,
    variance_threshold=0.85,
    n_runs=30,
    random_state=42
):
    df = pd.read_csv(csv_path)

    # feature selection
    feature_cols = df.columns.difference(["Country Name", "Country Code"])
    X = df[feature_cols]

    # imputazione SOLO per PCA/Hopkins
    X = X.fillna(X.mean())

    # normalizzazione
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # numero componenti per varianza cumulata
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cum_var, variance_threshold) + 1

    X_reduced = X_pca[:, :n_components]

    print(f"📉 PCA components selected: {n_components}")
    print(f"📊 Explained variance: {cum_var[n_components-1]:.3f}")

    # Hopkins ripetuta
    values = []
    for i in range(n_runs):
        H = hopkins_statistic(
            X_reduced,
            random_state=random_state + i
        )
        values.append(H)

    print("\n📊 Hopkins on PCA space")
    print(f"Mean: {np.mean(values):.4f}")
    print(f"Std:  {np.std(values):.4f}")
    print(f"Min:  {np.min(values):.4f}")
    print(f"Max:  {np.max(values):.4f}")

    return values, n_components

hopkins_values = run_hopkins_on_csv(
    csv_path="./data/preprocessing/final_dataset_cleaning_and_integration/final_data/clean/data_2017.csv",
    n_runs=30
)

hopkins_values, n_comp = hopkins_on_pca(
    csv_path="./data/preprocessing/final_dataset_cleaning_and_integration/final_data/clean/data_2017.csv",
    variance_threshold=0.85,
    n_runs=30
)