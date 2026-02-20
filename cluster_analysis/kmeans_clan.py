import pandas as pd
import glob
import os


path = "../data/SERIES/SERIES_k-means_4_20260220_175024"
files = sorted(glob.glob(os.path.join(path, "*.csv")))

all_data = []

for file in files:
    filename = os.path.basename(file)

    year = filename.split("_")[1]

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # sicurezza

    df["Year"] = year
    all_data.append(df[["Country Code", "Year", "cluster"]])

long_df = pd.concat(all_data, ignore_index=True)
long_df = long_df.drop_duplicates(subset=["Country Code", "Year"])


panel = long_df.pivot_table(
    index="Country Code",
    columns="Year",
    values="cluster",
    aggfunc="first"
)

panel = panel.reindex(sorted(panel.columns), axis=1)


def stability(row):
    row = row.dropna()
    if len(row) == 0:
        return float("nan")
    return row.value_counts().max() / len(row)


def transitions(row):
    row = row.dropna()
    return (row != row.shift()).sum() - 1


panel["stability"] = panel.apply(stability, axis=1)
panel["n_transitions"] = panel.apply(transitions, axis=1)


output_dir = "../data/series analysis/kmeans/"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "series_analysis_kmeans_4.csv")
panel.to_csv(output_path)
print("File panel salvato in:", output_path)
print("Shape finale:", panel.shape)


years = panel.columns[:-2]  # esclude stability e n_transitions
cluster_labels = sorted(long_df["cluster"].dropna().unique())

# Matrice contatori
transition_counts = pd.DataFrame(
    0,
    index=cluster_labels,
    columns=cluster_labels
)

# Conta tutte le transizioni anno anno
for i in range(len(years) - 1):
    y1 = years[i]
    y2 = years[i + 1]

    for c_from in cluster_labels:
        for c_to in cluster_labels:
            count = ((panel[y1] == c_from) & (panel[y2] == c_to)).sum()
            transition_counts.loc[c_from, c_to] += count

# Normalizzazione per riga probabilità
transition_matrix = transition_counts.div(
    transition_counts.sum(axis=1),
    axis=0
)

# Salvataggio matrice
transition_output_path = os.path.join(output_dir, "trans_matrix_kmeans.csv")
transition_matrix.to_csv(transition_output_path)
print("Matrice di transizione salvata in:", transition_output_path)

top10_transitions = panel["n_transitions"].sort_values(ascending=False).head(10)

# Distribuzione dei paesi per numero di spostamenti
transition_counts_summary = panel["n_transitions"].value_counts().sort_index()
transition_counts_summary = transition_counts_summary.rename_axis("n_transitions").reset_index(name="n_countries")

# Unisco le due sezioni in un unico CSV
stats_output_path = os.path.join(output_dir, "kmeans_stats.csv")

with open(stats_output_path, "w") as f:
    f.write("# Top 10 countries by number of transitions\n")
    top10_transitions.to_csv(f, header=["n_transitions"])
    f.write("\n# Distribution of transitions across countries\n")
    transition_counts_summary.to_csv(f, index=False)

print("Statistiche salvate in:", stats_output_path)

top_countries_per_cluster = {}

for cluster in cluster_labels:
    # Conta quante volte ogni paese appare in questo cluster
    cluster_data = long_df[long_df["cluster"] == cluster]
    counts = cluster_data["Country Code"].value_counts()

    # Prendi top 20 e salva in formato "Codice (conteggio)"
    top20_formatted = [f"{code} ({count})" for code, count in counts.head(20).items()]
    top_countries_per_cluster[cluster] = top20_formatted

# --- Creazione DataFrame per esportazione ---
# Troviamo la lunghezza massima tra i top20 dei cluster
max_len = max(len(lst) for lst in top_countries_per_cluster.values())

# Allunghiamo le liste con stringhe vuote se più corte di max_len
for cluster in top_countries_per_cluster:
    lst = top_countries_per_cluster[cluster]
    if len(lst) < max_len:
        lst.extend([""] * (max_len - len(lst)))

top_countries_df = pd.DataFrame(top_countries_per_cluster)

# Salvataggio CSV
top_countries_path = os.path.join(output_dir, "top_countries_per_cluster.csv")
top_countries_df.to_csv(top_countries_path, index=False)
print("Top 20 country codes per cluster salvati in:", top_countries_path)
