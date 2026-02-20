import shutil

import pandas as pd
import os
from glob import glob
import re
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def compute_correlation_matrix(input_csv: str, output_csv: str = None):
    """
    Calcola la matrice di correlazione tra tutti gli attributi numerici
    di un dataset CSV, escludendo colonne non numeriche come 'Country' e 'Country Code'.

    Parameters
    ----------
    input_csv : str
        Percorso del CSV di input.
    output_csv : str, optional
        Percorso per salvare la matrice di correlazione (default: None).

    Returns
    -------
    pd.DataFrame
        DataFrame contenente la matrice di correlazione.
    """
    # Leggi CSV
    df = pd.read_csv(input_csv)

    # Identifica colonne numeriche (tutti gli attributi)
    numeric_cols = df.select_dtypes(include="number").columns

    # Calcola matrice di correlazione
    corr_matrix = df[numeric_cols].corr(method="pearson")

    # Salva CSV se richiesto
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        corr_matrix.to_csv(output_csv)
        print(f"Matrice di correlazione salvata in: {output_csv}")

    return corr_matrix


def extract_high_correlations(input_folder, output_csv, threshold=0.85):
    """
    Estrae coppie di attributi con correlazione maggiore di 'threshold'
    da tutte le correlation matrices presenti nella cartella.

    Parameters
    ----------
    input_folder : str
        Cartella contenente file CSV delle correlation matrices
        con nomi "corr_matrix_{year}.csv".
    output_csv : str
        Percorso per salvare il CSV finale.
    threshold : float
        Soglia minima di correlazione per includere una coppia.
    """
    # Dizionario: chiave = (attr1, attr2), valore = lista di tuple (anno, corr)
    high_corr_dict = {}

    # Lista CSV nella cartella
    csv_files = glob(os.path.join(input_folder, "corr_matrix_*.csv"))

    for csv_file in csv_files:
        # Estrai anno dal nome file
        match = re.search(r"corr_matrix_(\d{4})\.csv", os.path.basename(csv_file))
        if not match:
            print(f"Anno non trovato in {csv_file}, skip")
            continue
        year = int(match.group(1))

        # Leggi matrice di correlazione
        corr_df = pd.read_csv(csv_file, index_col=0)

        # Cicla solo sulla metà superiore della matrice per non duplicare coppie
        for i, attr1 in enumerate(corr_df.columns):
            for j, attr2 in enumerate(corr_df.columns):
                if j <= i:
                    continue  # metà superiore solo
                corr_value = corr_df.loc[attr1, attr2]
                if abs(corr_value) >= threshold:
                    key = tuple(sorted([attr1, attr2]))
                    if key not in high_corr_dict:
                        high_corr_dict[key] = []
                    high_corr_dict[key].append((year, corr_value))

    # Trasforma in DataFrame finale
    rows = []
    for (attr1, attr2), year_corr_list in high_corr_dict.items():
        year_corr_str = "; ".join([f"{y}: {c:.3f}" for y, c in sorted(year_corr_list)])
        count_years = len(year_corr_list)
        rows.append({
            "Attribute1": attr1,
            "Attribute2": attr2,
            "Years & Correlations": year_corr_str,
            "CountYears": count_years
        })

    final_df = pd.DataFrame(rows)
    final_df.sort_values(by=["Attribute1", "Attribute2"], inplace=True)

    # Salva CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"CSV finale salvato in {output_csv}")

    return final_df


def filter_columns_by_list(data_csv, keep_list_csv, output_csv=None):
    """
    Mantiene solo le colonne presenti nella lista del file keep_list_csv.

    Parameters
    ----------
    data_csv : str
        Path del CSV con i dati originali.
    keep_list_csv : str
        Path del CSV con una colonna contenente i nomi degli attributi da tenere.
    output_csv : str, optional
        Path per salvare il CSV filtrato. Se None, non salva.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrato.
    """
    # Carica dati
    df_data = pd.read_csv(data_csv)

    # Carica lista di colonne da mantenere
    df_list = pd.read_csv(keep_list_csv)
    keep_cols = df_list.iloc[:, 0].tolist()  # Prende la prima colonna

    # Mantieni solo colonne che sono in keep_cols
    cols_to_keep = [col for col in df_data.columns if col in keep_cols]
    df_filtered = df_data[cols_to_keep].copy()

    # Salva se richiesto
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_filtered.to_csv(output_csv, index=False)
        print(f"CSV filtrato salvato in {output_csv}")

    return df_filtered

def fill_missing_vals(input_csv, output_csv=None, exclude_cols=None, measure="Mean"):

    if exclude_cols is None:
        exclude_cols = ["Country Name", "Country Code"]

    df = pd.read_csv(input_csv)

    numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]


    # Sostituisci missing values con la MEDIA per colonna
    if measure == "Mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    else:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"CSV normalizzato salvato in {output_csv}")

    return df


def create_median_csv(folder_path, attributes, output_csv="median_WHR.csv", country_col="Country Name"):
    """
    Crea un CSV con le mediane per paese e attributo usando tutti i CSV in una cartella.

    Parameters
    ----------
    folder_path : str
        Percorso della cartella con i CSV annuali.
    attributes : list of str
        Lista delle colonne numeriche da considerare.
    output_csv : str
        Percorso e nome del CSV di output.
    country_col : str
        Nome della colonna che identifica i paesi nei CSV.
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        raise ValueError("Nessun CSV trovato nella cartella specificata.")

    # Lista per concatenare tutti i dati
    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)
        # Seleziona solo country + attributi
        cols = [country_col] + [attr for attr in attributes if attr in df.columns]
        all_data.append(df[cols])

    # Concatenazione di tutti gli anni
    combined_df = pd.concat(all_data, ignore_index=True)

    # Calcolo della mediana per paese e attributo
    median_df = combined_df.groupby(country_col).median(numeric_only=True).reset_index()

    for col in median_df.columns:
        if col != country_col:
            if median_df[col].isna().any():
                col_median = median_df[col].median()
                median_df[col] = median_df[col].fillna(col_median)  # <-- assegnamento diretto
                print(f"Attenzione: missing values in '{col}' imputati con mediana {col_median}")
    # Salvataggio
    median_df.to_csv(output_csv, index=False)
    print(f"File median_WHR creato in: {output_csv}")

    return median_df

def zscore_and_fill_csv(input_csv, median_csv="median_WHR.csv", output_csv=None,
                        exclude_cols_fill=None, exclude_cols_zscore=None,
                        measure="Mean", jitter=1e-6):
    """
    Normalizza tutte le colonne numeriche di un CSV usando z-score,
    e riempie i missing values:
        - Per le colonne in exclude_cols_fill usa la mediana storica per paese
        - Per le altre colonne usa media o mediana globale

    Parameters
    ----------
    input_csv : str
        Percorso del CSV di input.
    median_csv : str
        Percorso CSV con le mediane per paese e attributo.
    output_csv : str, optional
        Percorso per salvare il CSV normalizzato. Se None, non salva.
    exclude_cols_fill : list of str, optional
        Colonne da escludere dal fill globale (useremo mediana per paese).
    exclude_cols_zscore : list of str, optional
        Colonne da escludere dallo z-score.
    measure : str
        "Mean" o "Median" per le colonne non in exclude_cols_fill.
    jitter : float
        Ampiezza del jitter da aggiungere ai valori imputati (minimo).
    """

    if exclude_cols_fill is None:
        exclude_cols_fill = ["Country Name", "Country Code"]

    if exclude_cols_zscore is None:
        exclude_cols_zscore = ["Country Name", "Country Code"]

    # Leggi CSV dei dati
    df = pd.read_csv(input_csv)

    # Leggi CSV mediane
    median_df = pd.read_csv(median_csv)
    median_df.set_index("Country Name", inplace=True)  # per accesso veloce

    # Colonne da imputare globalmente
    cols_global_fill = [col for col in df.columns if
                        col not in exclude_cols_fill and pd.api.types.is_numeric_dtype(df[col])]

    # Colonne da imputare con mediana paese
    cols_country_fill = [col for col in exclude_cols_fill if
                         col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    # --- 1️⃣ Fill globale per le colonne "normali" ---
    if measure == "Mean":
        df[cols_global_fill] = df[cols_global_fill].fillna(df[cols_global_fill].mean())
    else:
        df[cols_global_fill] = df[cols_global_fill].fillna(df[cols_global_fill].median())

    # --- 2️⃣ Fill per paese usando mediane storiche ---
    for col in df.columns:
        if col not in cols_global_fill and pd.api.types.is_numeric_dtype(df[col]):
            # Scorri le righe
            for idx, row in df[df[col].isna()].iterrows():
                country = row["Country Name"]
                if country in median_df.index and col in median_df.columns:
                    median_val = median_df.loc[country, col]
                    # aggiungi jitter minimo
                    jitter_val = median_val * jitter * np.random.randn()
                    df.at[idx, col] = median_val + jitter_val
                else:
                    # fallback: mediana globale
                    fallback = df[col].median()
                    df.at[idx, col] = fallback + fallback * jitter * np.random.randn()

    # --- 3️⃣ Z-score ---
    cols_to_scale = [col for col in df.columns if col not in exclude_cols_zscore]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # --- 4️⃣ Salva se richiesto ---
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"CSV normalizzato salvato in {output_csv}")

    return df

if __name__ == "__main__":

    # for year in range(2005,2023):
    #     input_path = f"data/preprocessing/final_dataset_cleaning_and_integration/merged_data/2-no_empty_countries/data/merged_{year}.csv"
    #     output_path = f"data/preprocessing/final_dataset_cleaning_and_integration/merged_data/3-corr_analysis/matrices/corr_matrix_{year}.csv"
    #     corr_df = compute_correlation_matrix(input_path, output_path)

    exclude_cols = ["happiness_score",
        "social_support",
        "healthy_life_expectancy_at_birth",
        "freedom_to_make_life_choices",
        "generosity",
        "perceptions_of_corruption",
        "positive_affect",
        "negative_affect",
        ]

    input_folder = "data/preprocessing/final_dataset_cleaning_and_integration/merged_data/3-corr_analysis/matrices"
    output_csv = "data/preprocessing/final_dataset_cleaning_and_integration/merged_data/3-corr_analysis/correlation.csv"
    #extract_high_correlations(input_folder, output_csv, threshold=0.85)
    median_csv = "data/preprocessing/final_dataset_cleaning_and_integration/final_data/median_WHR.csv"
    median_folder = "data/preprocessing/final_dataset_cleaning_and_integration/final_data/original"

    # for year in range(2005,2023):
    #     data_csv = f"data/preprocessing/final_dataset_cleaning_and_integration/merged_data/2-no_empty_countries/data/merged_{year}.csv"
    #     output_csv = f"data/preprocessing/final_dataset_cleaning_and_integration/final_data/original/final_data_{year}.csv"
    #     keep_list_csv = "data/preprocessing/final_dataset_cleaning_and_integration/merged_data/3-corr_analysis/FINAL_ATTRIBUTES_LIST.csv"
    #     filter_columns_by_list(data_csv, keep_list_csv, output_csv)

    create_median_csv(median_folder,exclude_cols,median_csv)

    for year in range(2005, 2023):
        output_csv = f"data/preprocessing/final_dataset_cleaning_and_integration/final_data/clean/data_{year}.csv"
        input_csv = f"data/preprocessing/final_dataset_cleaning_and_integration/final_data/original/final_data_{year}.csv"
        #fill_missing_vals(input_csv, output_csv, exclude_cols, "Median")
        zscore_and_fill_csv(input_csv, median_csv, output_csv, exclude_cols, None, "Median")

    # salviamo i file nella cartella final_dataset
    source_folder = Path("data/preprocessing/final_dataset_cleaning_and_integration/final_data/clean")
    destination_folder = Path("data/final_dataset")
    destination_folder.mkdir(parents=True, exist_ok=True)

    for file in source_folder.glob("*.csv"):  # tutti i CSV
        shutil.copy2(file, destination_folder / file.name)
        print(f"File copiato: {file.name}")