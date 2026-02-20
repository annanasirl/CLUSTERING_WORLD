import pandas as pd
import glob
import os

# --- Percorso ai file HIER ---
path = "../data/SERIES/SERIES_hierarchical_2_20260220_175147"
files = sorted(glob.glob(os.path.join(path, "*.csv")))

all_data = []

# Anni in cui invertire i cluster
invert_years = [2009, 2010, 2013, 2016] + list(range(2018, 2023))

for file in files:
    filename = os.path.basename(file)
    year = int(filename.split("_")[1])  # converto in int

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    df["Year"] = year



    # --- Inversione cluster per gli anni selezionati ---
    if year in invert_years:
        df["cluster"] = df["cluster"].map({0: 1, 1: 0})

    all_data.append(df[["Country Code", "Year", "cluster"]])

# --- Concatenazione dati ---
long_df = pd.concat(all_data, ignore_index=True)
long_df = long_df.drop_duplicates(subset=["Country Code", "Year"])

# --- Pivot table panel ---
panel = long_df.pivot_table(
    index="Country Code",
    columns="Year",
    values="cluster",
    aggfunc="first"
)
panel = panel.reindex(sorted(panel.columns), axis=1)

# --- Funzioni metriche ---
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

# --- Salvataggio CSV panel ---
output_dir = "../data/series analysis/hier_meta_inverted/"
os.makedirs(output_dir, exist_ok=True)

panel_path = os.path.join(output_dir, "series_analysis_hier_meta_inverted.csv")
panel.to_csv(panel_path)
print("Panel salvato in:", panel_path)
print("Shape finale:", panel.shape)

# --- Matrice di transizione ---
years = panel.columns[:-2]  # esclude stability e n_transitions
cluster_labels = sorted(long_df["cluster"].dropna().unique())

transition_counts = pd.DataFrame(
    0,
    index=cluster_labels,
    columns=cluster_labels
)

for i in range(len(years) - 1):
    y1 = years[i]
    y2 = years[i + 1]
    for c_from in cluster_labels:
        for c_to in cluster_labels:
            count = ((panel[y1] == c_from) & (panel[y2] == c_to)).sum()
            transition_counts.loc[c_from, c_to] += count

transition_matrix = transition_counts.div(
    transition_counts.sum(axis=1),
    axis=0
)

transition_matrix_path = os.path.join(output_dir, "trans_matrix_hier_meta_inverted.csv")
transition_matrix.to_csv(transition_matrix_path)
print("Matrice di transizione salvata in:", transition_matrix_path)

# --- Statistiche paesi ---
top10_transitions = panel["n_transitions"].sort_values(ascending=False).head(10)

transition_counts_summary = panel["n_transitions"].value_counts().sort_index()
transition_counts_summary = transition_counts_summary.rename_axis("n_transitions").reset_index(name="n_countries")

stats_path = os.path.join(output_dir, "hier_meta_inverted_stats.csv")
with open(stats_path, "w") as f:
    f.write("# Top 10 countries by number of transitions\n")
    top10_transitions.to_csv(f, header=["n_transitions"])
    f.write("\n# Distribution of transitions across countries\n")
    transition_counts_summary.to_csv(f, index=False)
print("Statistiche salvate in:", stats_path)

# --- Top 20 countries per cluster ---
top_countries_per_cluster = {}
for cluster in cluster_labels:
    cluster_data = long_df[long_df["cluster"] == cluster]
    counts = cluster_data["Country Code"].value_counts()
    top20_formatted = [f"{code} ({count})" for code, count in counts.head(20).items()]
    top_countries_per_cluster[cluster] = top20_formatted

# Uniformo le lunghezze
max_len = max(len(lst) for lst in top_countries_per_cluster.values())
for cluster in top_countries_per_cluster:
    lst = top_countries_per_cluster[cluster]
    if len(lst) < max_len:
        lst.extend([""] * (max_len - len(lst)))

top_countries_df = pd.DataFrame(top_countries_per_cluster)
top_countries_path = os.path.join(output_dir, "top_countries_per_cluster_meta_inverted.csv")
top_countries_df.to_csv(top_countries_path, index=False)
print("Top 20 country codes per cluster salvati in:", top_countries_path)