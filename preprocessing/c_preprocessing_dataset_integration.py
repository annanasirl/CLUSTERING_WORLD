import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from glob import glob
import os
import re
from collections import defaultdict


def build_yearly_datasets_WDI(
    wdi_csv_path: str,
    indicators_csv_path: str,
    output_dir: str,
    start_year: int = 2002,
    end_year: int = 2024,
    indicator_column: str = "Indicator Name"
):

    # Load datasets
    wdi = pd.read_csv(wdi_csv_path, low_memory=False)
    indicators = pd.read_csv(indicators_csv_path)

    selected_indicators = indicators[indicator_column].unique()

    # Filter selected indicators
    wdi = wdi[wdi["Indicator Name"].isin(selected_indicators)]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        year_str = str(year)

        if year_str not in wdi.columns:
            print(f"Year {year} not found, skipping.")
            continue

        df_year = wdi[
            ["Country Name", "Country Code", "Indicator Name", year_str]
        ].rename(columns={year_str: "Value"})

        # 🔑 Force numeric conversion
        df_year["Value"] = pd.to_numeric(df_year["Value"], errors="coerce")

        # Pivot: one row per country, one column per indicator
        df_year = df_year.pivot_table(
            index=["Country Name", "Country Code"],
            columns="Indicator Name",
            values="Value",
            aggfunc="first"   # IMPORTANT
        ).reset_index()

        output_file = output_path / f"WDI_{year}.csv"
        df_year.to_csv(output_file, index=False)

        print(f"Saved {output_file}")

import pandas as pd
from pathlib import Path

def split_csv_by_year(input_csv: str, output_dir: str, country_col='Country', year_col='Year'):
    """
    Divide un CSV in più file, uno per ogni anno.

    Parameters
    ----------
    input_csv : str
        Percorso al CSV originale.
    output_dir : str
        Cartella dove salvare i CSV per anno.
    country_col : str
        Nome della colonna che contiene i paesi.
    year_col : str
        Nome della colonna che contiene gli anni.
    """
    # Crea la cartella di output se non esiste
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Carica CSV
    df = pd.read_csv(input_csv)

    # Trova tutti gli anni unici
    years = df[year_col].unique()

    # Per ogni anno, salva un CSV
    for yr in years:
        df_year = df[df[year_col] == yr].copy()
        # Rimuovo la colonna anno
        df_year = df_year.drop(columns=[year_col])
        # Salva CSV
        year_file = output_path / f"data_{yr}.csv"
        df_year.to_csv(year_file, index=False)
        print(f"Saved: {year_file}")

# Esempio di utilizzo
# split_csv_by_year("dataset_completo.csv", "output_per_anno")


def clean_and_normalize_csv(file_path: str, output_path: str = None):
    """
    - Sostituisce i valori mancanti con la MEDIA per colonna
    - Normalizza tutte le colonne numeriche (z-score)
    - Salva il CSV normalizzato se output_path è specificato
    - Ritorna il DataFrame normalizzato
    """
    df = pd.read_csv(file_path)

    # Identifica colonne numeriche
    numeric_cols = df.select_dtypes(include=["number"]).columns

    # Sostituisci missing values con la MEDIA per colonna
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Normalizza con z-score (colonna per colonna)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Salva se richiesto
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Normalized CSV saved to {output_path}")

    return df


def merge_csv_folders_per_year(
        folder1,
        folder2,
        output_dir="merged_per_year",
        country_col1="Country Code",
        country_col2="Country Code"
):
    """
    Merge CSV tra due cartelle, accoppiando i file per anno
    estratto dal nome del file (es. dati_2020.csv).

    Parameters
    ----------
    folder1 : str
        Prima cartella di CSV.
    folder2 : str
        Seconda cartella di CSV.
    output_dir : str
        Cartella dove salvare i CSV finali.
    country_col1 : str
        Nome della colonna "Country Code" nei CSV di folder1.
    country_col2 : str
        Nome della colonna "Country Code" nei CSV di folder2.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lista dei file nelle due cartelle
    files1 = glob(os.path.join(folder1, "*.csv"))
    files2 = glob(os.path.join(folder2, "*.csv"))

    # Crea un dizionario anno -> file
    def map_files_by_year(file_list):
        mapping = {}
        for f in file_list:
            match = re.search(r"(20\d{2})", os.path.basename(f))
            if match:
                year = int(match.group(1))
                mapping[year] = f
        return mapping

    mapping1 = map_files_by_year(files1)
    mapping2 = map_files_by_year(files2)

    # Trova anni comuni
    common_years = set(mapping1.keys()) & set(mapping2.keys())
    if not common_years:
        print("Nessun anno comune trovato tra le due cartelle.")
        return

    for year in sorted(common_years):
        print(f"Elaboro anno {year}")

        df1 = pd.read_csv(mapping1[year])
        df2 = pd.read_csv(mapping2[year])

        # Pulisce eventuali spazi nelle colonne
        df1.columns = df1.columns.str.strip()
        df2.columns = df2.columns.str.strip()

        # Controlla che le colonne country esistano
        if country_col1 not in df1.columns:
            raise ValueError(f"'{country_col1}' mancante in {mapping1[year]}")
        if country_col2 not in df2.columns:
            raise ValueError(f"'{country_col2}' mancante in {mapping2[year]}")

        # Merge sui paesi
        df_merged = pd.merge(
            df1, df2,
            left_on=country_col1,
            right_on=country_col2,
            how="outer"
        )

        # Elimina la colonna duplicata del country code della seconda cartella
        if country_col1 != country_col2:
            df_merged.drop(columns=[country_col2], inplace=True)

        # Salva CSV finale
        output_file = os.path.join(output_dir, f"merged_{year}.csv")
        df_merged.to_csv(output_file, index=False)
        print(f"Salvato: {output_file}")

    print(f"Tutti i file finali salvati in '{output_dir}'")


ATTRIBUTI = [
    "happiness_score",
    "social_support",
    "healthy_life_expectancy_at_birth",
    "freedom_to_make_life_choices",
    "generosity",
    "perceptions_of_corruption",
    "positive_affect",
    "negative_affect"
]

def analizza_valori_vuoti(
    input_dir,
    output_dir="output_analysis",
    col_country="Country Name",
    col_country_code="Country Code"
):
    os.makedirs(output_dir, exist_ok=True)

    risultati_annuali = []
    contatori_paesi = defaultdict(lambda: {
        "country": None,
        "country_code": None,
        "times_all_empty": 0,
        "times_partially_empty": 0
    })

    for file in glob(os.path.join(input_dir, "*.csv")):
        match = re.search(r"(20\d{2})", os.path.basename(file))
        if not match:
            continue

        year = int(match.group(1))
        print(f"➡️ Analizzo anno {year}")

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        count_all_empty = 0
        count_partially_empty = 0

        for _, row in df.iterrows():
            valori = row[ATTRIBUTI]

            all_nan = valori.isna().all()
            any_nan = valori.isna().any()

            country = row[col_country]
            code = row[col_country_code]

            contatori_paesi[code]["country"] = country
            contatori_paesi[code]["country_code"] = code

            if all_nan:
                count_all_empty += 1
                contatori_paesi[code]["times_all_empty"] += 1

            if any_nan:
                count_partially_empty += 1
                contatori_paesi[code]["times_partially_empty"] += 1

        risultati_annuali.append({
            "year": year,
            "all_empty_countries": count_all_empty,
            "partially_empty_countries": count_partially_empty
        })

    # ---- empty_values.csv (NUMERI) ----
    df_empty = pd.DataFrame(risultati_annuali).sort_values("year")
    df_empty.to_csv(
        os.path.join(output_dir, "empty_values.csv"),
        index=False
    )

    # ---- surveyed_countries.csv ----
    df_paesi = pd.DataFrame(contatori_paesi.values()).sort_values("country")
    df_paesi.to_csv(
        os.path.join(output_dir, "surveyed_countries.csv"),
        index=False
    )

    print("\n✅ File creati con successo:")
    print(" - empty_values.csv (conteggi)")
    print(" - surveyed_countries.csv")

def rimuovi_paesi_tutti_vuoti(
    surveyed_file,
    cartella_csv_annuali,
    output_dir="csv_annuali_puliti",
    col_country_code="Country Code"
):
    os.makedirs(output_dir, exist_ok=True)

    # Leggi surveyed_countries.csv
    df_surveyed = pd.read_csv(surveyed_file)
    df_surveyed.columns = df_surveyed.columns.str.strip()

    # Paesi da rimuovere: times_all_empty == 18
    paesi_da_rimuovere = df_surveyed.loc[
        df_surveyed["times_all_empty"] == 18, "country_code"
    ].tolist()

    print(f"Totale paesi da rimuovere: {len(paesi_da_rimuovere)}")

    # Cicla su tutti i CSV annuali
    for file_csv in glob(os.path.join(cartella_csv_annuali, "*.csv")):
        print(f"➡️ Elaboro {file_csv}")

        df = pd.read_csv(file_csv)
        df.columns = df.columns.str.strip()

        # --- ELIMINA COLONNA 'country' SE ESISTE ---
        if "country" in df.columns:
            df.drop(columns=["country"], inplace=True)

        # Filtra: mantiene solo i paesi NON nella lista da rimuovere
        df_filtrato = df[~df[col_country_code].isin(paesi_da_rimuovere)]

        # Salva nella cartella di output
        output_file = os.path.join(output_dir, os.path.basename(file_csv))
        df_filtrato.to_csv(output_file, index=False)

    print(f"\n✅ Tutti i CSV annuali puliti salvati in '{output_dir}'")


if __name__ == "__main__":
    wdi_csv_path = "../data/og_dataset/WDI/WDICSV.csv"
    wdi_indicators_csv_path = "../data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/final_attributes_chosen.csv"
    wdi_output_dir = "../data/preprocessing/final_dataset_cleaning_and_integration/wdi_data/og"
    whr_output_dir = "../data/preprocessing/final_dataset_cleaning_and_integration/whr_data/og"
    # wdi_output_dir_c = "data/preprocessing/final_dataset_cleaning_and_integration/wdi_data/clean"
    # whr_output_dir_c = "data/preprocessing/final_dataset_cleaning_and_integration/whr_data/clean"

    start_year = 2005
    end_year = 2024
    indicator_column = "Indicator Name"
    whr_file = "../data/preprocessing/coarse_whr_dim_r/world_happiness_all_years.csv"
    merged_file = "../data/preprocessing/final_dataset_cleaning_and_integration/merged_data/1-before_fine_dim_red"

    build_yearly_datasets_WDI( wdi_csv_path, wdi_indicators_csv_path, wdi_output_dir, start_year, end_year, indicator_column)

    split_csv_by_year(whr_file, whr_output_dir, country_col = 'country', year_col = 'year')

    # for year in range(2005, 2023):  # 2002 fino a 2024 incluso
    #     wdi_input_file = f"data/preprocessing/final_dataset_cleaning_and_integration/wdi_data/og/WDI_{year}.csv"
    #     wdi_output_file = f"data/preprocessing/final_dataset_cleaning_and_integration/wdi_data/clean/WDI_{year}_normalized.csv"
    #     clean_and_normalize_csv(wdi_input_file, wdi_output_file)
    #     whr_input_file = f"data/preprocessing/final_dataset_cleaning_and_integration/whr_data/og/data_{year}.csv"
    #     whr_output_file = f"data/preprocessing/final_dataset_cleaning_and_integration/whr_data/clean/WHR_{year}_normalized.csv"
    #     clean_and_normalize_csv(whr_input_file, whr_output_file)
    #
    merge_csv_folders_per_year(wdi_output_dir, whr_output_dir, merged_file, "Country Code", "cntry_code")

    # analizza_valori_vuoti(
    #      merged_file,
    #      output_dir="data/preprocessing/final_dataset_cleaning_and_integration/merged_data/2-no_empty_countries",
    #      col_country="Country Name",
    #      col_country_code="Country Code"
    # )
    #
    surveyed_file = "../data/preprocessing/final_dataset_cleaning_and_integration/merged_data/2-no_empty_countries/surveyed_countries.csv"
    clean_data = "data/preprocessing/final_dataset_cleaning_and_integration/merged_data/2-no_empty_countries/data"

    rimuovi_paesi_tutti_vuoti(
        surveyed_file,
        merged_file,
        clean_data,
        col_country_code="Country Code"
    )

