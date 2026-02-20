import pandas as pd
import os
from glob import glob
from sklearn.preprocessing import StandardScaler
from b_preprocessing_exploration_pca import run_pca

def esporta_conteggi_csv(file_input, nome_colonna, file_output):
    # Legge il CSV
    df = pd.read_csv(file_input)

    # Controllo che la colonna esista
    if nome_colonna not in df.columns:
        raise ValueError(f"La colonna '{nome_colonna}' non esiste nel file CSV")

    # Conteggio valori distinti (inclusi NaN)
    conteggi = df[nome_colonna].value_counts(dropna=False)

    # Trasformo in DataFrame con due colonne
    risultato = conteggi.reset_index()
    risultato.columns = [nome_colonna, "occorrenze"]

    # Salvo il nuovo CSV
    risultato.to_csv(file_output, index=False)
    print(f"File '{file_output}' creato con successo!")

    return risultato

def taglia_categoria(file_input, nome_colonna, file_output):
    # Legge il CSV
    df = pd.read_csv(file_input)

    # Controllo colonna
    if nome_colonna not in df.columns:
        raise ValueError(f"La colonna '{nome_colonna}' non esiste nel file CSV")

    # Funzione di pulizia
    def estrai_prima_parte(valore):
        if pd.isna(valore):  # ignora valori vuoti
            return valore
        val_str = str(valore).strip()  # rimuove spazi iniziali/finali
        return val_str.split(":", 1)[0] if ":" in val_str else val_str

    # Applica la funzione
    df[nome_colonna] = df[nome_colonna].apply(estrai_prima_parte)

    # Salva il nuovo CSV
    df.to_csv(file_output, index=False)
    print(f"File '{file_output}' creato con la colonna '{nome_colonna}' modificata!")

    return df


def raggruppa_topic(
    file_input,
    colonna,
    file_output,
    separatore="; "
):
    df = pd.read_csv(file_input)

    aggregato = {}

    for valore in df[colonna]:
        if pd.isna(valore):
            continue

        testo = str(valore).strip()

        if ":" not in testo:
            continue

        # split SOLO sul primo :
        principale, resto = testo.split(":", 1)

        principale = principale.strip()
        resto = resto.strip()

        if not principale or not resto:
            continue

        aggregato.setdefault(principale, [])

        if resto not in aggregato[principale]:
            aggregato[principale].append(resto)

    risultato = pd.DataFrame({
        colonna: aggregato.keys(),
        "valori": [separatore.join(v) for v in aggregato.values()]
    })

    risultato.to_csv(file_output, index=False)
    print(f"File '{file_output}' creato correttamente!")

def generate_indicator_coverage(input_folder, output_folder, years=None):

    # crea cartella output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # lista tutti i file CSV nella cartella input
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    # se non specificati, consideriamo le colonne 2002-2024
    if years is None:
        years = [str(y) for y in range(2005, 2025)]

    for file in csv_files:
        filepath = os.path.join(input_folder, file)
        df = pd.read_csv(filepath)

        # selezioniamo solo colonne indicatori + anni
        df_subset = df[['Indicator Name'] + years]

        # raggruppiamo per indicatore
        grouped = df_subset.groupby('Indicator Name')

        # numero totale di celle per indicatore
        total_values = grouped.size() * len(years)

        # numero di valori null per indicatore
        null_values = grouped[years].apply(lambda x: x.isnull().sum().sum())

        # creiamo DataFrame di summary
        summary = pd.DataFrame({
            'Indicator Name': total_values.index,
            'total_values': total_values.values,
            'null_values': null_values.values
        })

        # calcola percentuale coverage
        summary['coverage_percent'] = 100 * (summary['total_values'] - summary['null_values']) / summary['total_values']

        # ordina per coverage decrescente
        summary = summary.sort_values(by='coverage_percent', ascending=False)

        # salva CSV di output
        output_path = os.path.join(output_folder, f"coverage_{file}")
        summary.to_csv(output_path, index=False)

        print(f"Processed {file}, saved coverage summary to {output_path}")

def split_csv_per_attributo(file_input, colonna, output_dir="output_files", chunk_size=100000):
    """
    Legge un CSV grande a chunk e crea un CSV separato per ogni valore unico della colonna indicata.

    Parametri:
    - file_input: percorso del CSV grande
    - colonna: colonna degli attributi
    - output_dir: cartella dove salvare i CSV
    - chunk_size: numero di righe lette per volta
    """
    # Crea la cartella se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Dizionario che memorizza i nomi dei file creati
    file_handles = {}

    # Legge il CSV a chunk
    for chunk in pd.read_csv(file_input, chunksize=chunk_size):
        # Pulisce la colonna: prende solo la parte prima del primo :
        def estrai_prima_parte(valore):
            if pd.isna(valore):
                return "NA"  # gestisce valori vuoti
            testo = str(valore).strip()
            return testo.split(":", 1)[0] if ":" in testo else testo

        chunk[colonna] = chunk[colonna].apply(estrai_prima_parte)

        # Raggruppa per valore della colonna
        for valore_unico, df_gruppo in chunk.groupby(colonna):
            # Nome file sicuro
            nome_file = os.path.join(output_dir, f"{valore_unico.replace('/', '_').replace(' ', '_')}.csv")

            # Se il file esiste già, append senza header
            if os.path.exists(nome_file):
                df_gruppo.to_csv(nome_file, index=False, mode="a", header=False)
            else:
                df_gruppo.to_csv(nome_file, index=False, mode="w", header=True)

    print(f"Tutti i file creati nella cartella '{output_dir}'.")

def separa_per_macrocategoria(
    wdi_file,
    indicatori_dir,
    colonna_codici_macrocategoria,
    colonna_codici_wdi,
    output_dir="WDICSV_by_category",
    chunk_size=100000
):
    """
    wdi_file: path del file WDICSV.csv con tutti i dati
    indicatori_dir: cartella con i file dei codici degli indicatori per ogni macrocategoria
    colonna_codici_macrocategoria: nome della colonna dei codici negli indicatori
    colonna_codici_wdi: nome della colonna dei codici in WDICSV
    output_dir: cartella dove salvare i CSV filtrati
    chunk_size: numero di righe lette per volta
    """

    os.makedirs(output_dir, exist_ok=True)

    # Legge tutti i file delle macrocategorie
    macrocategorie_files = glob(os.path.join(indicatori_dir, "*.csv"))

    # Dizionario: macrocategoria -> set di codici indicatori
    indicatori_per_categoria = {}
    for file in macrocategorie_files:
        nome_categoria = os.path.splitext(os.path.basename(file))[0]
        df_ind = pd.read_csv(file)
        codici = df_ind[colonna_codici_macrocategoria].dropna().unique().tolist()
        indicatori_per_categoria[nome_categoria] = set(codici)

    print("Categorie trovate:", list(indicatori_per_categoria.keys()))

    # Legge WDICSV a chunk e filtra per categoria
    for chunk in pd.read_csv(wdi_file, chunksize=chunk_size):
        for categoria, codici in indicatori_per_categoria.items():
            df_filtrato = chunk[chunk[colonna_codici_wdi].isin(codici)]
            if not df_filtrato.empty:
                nome_file = os.path.join(output_dir, f"{categoria}.csv")
                if os.path.exists(nome_file):
                    df_filtrato.to_csv(nome_file, index=False, mode="a", header=False)
                else:
                    df_filtrato.to_csv(nome_file, index=False, mode="w", header=True)

    print(f"Tutti i file delle macrocategorie creati in '{output_dir}'.")

def prepare_pca_dataset(
    indicators_csv_path,
    data_csv_path,
    output_csv_path,
    year,
    indicator_col="Indicator Name",
    country_col="Country Name",
    imputation="mean"
):
    """
    Prepares a country x indicator dataset ready for PCA for a specified year.
    This version also standardizes the indicators (mean=0, std=1).

    Parameters
    ----------
    indicators_csv_path : str
        Path to CSV containing the list of selected indicators.
    data_csv_path : str
        Path to CSV containing WDI data for a macro-category.
    output_csv_path : str
        Path where the PCA-ready CSV will be saved.
    year : str
        Year to be used for the cross-sectional snapshot (e.g., '2019').
    indicator_col : str
        Column name for indicator names (default: 'Indicator Name').
    country_col : str
        Column name for country names (default: 'Country Name').
    imputation : str
        'mean' or 'median' for missing value imputation (default: 'median').
    """

    # Load indicator list
    indicators_df = pd.read_csv(indicators_csv_path)
    indicators = indicators_df[indicator_col].unique()

    # Load data
    data_df = pd.read_csv(data_csv_path)

    # Filter indicators and year
    filtered_df = data_df[
        data_df[indicator_col].isin(indicators)
    ][[country_col, indicator_col, year]].copy()

    # Convert values to numeric, coerce errors to NaN
    filtered_df[year] = pd.to_numeric(filtered_df[year], errors='coerce')

    # Pivot to Country x Indicator
    pca_df = filtered_df.pivot(
        index=country_col,
        columns=indicator_col,
        values=year
    )

    # Impute missing values
    if imputation == "mean":
        pca_df = pca_df.fillna(pca_df.mean())
    elif imputation == "median":
        pca_df = pca_df.fillna(pca_df.median())
    else:
        raise ValueError("imputation must be 'mean' or 'median'")

    # Standardize indicators
    scaler = StandardScaler()
    pca_scaled = pd.DataFrame(
        scaler.fit_transform(pca_df),
        index=pca_df.index,
        columns=pca_df.columns
    )

    # Save PCA-ready standardized dataset
    pca_scaled.to_csv(output_csv_path)
    print(f"PCA-ready standardized dataset for {year} saved to: {output_csv_path}")
    print(f"Shape: {pca_scaled.shape}")


if __name__ == "__main__":
    originale = "data/og_dataset/WDI/WDISeries.csv"
    og_data = "data/preprocessing/coarse_wdi_dim_red/1-cut_before_2005/WDICSV_from_2005.csv"
    colonna = "Topic"
    count = "data/preprocessing/coarse_wdi_dim_red/2-build_macrocategories/0-numero_indicatori_per_topic.csv"
    only_indicatori = "data/preprocessing/coarse_wdi_dim_red/2-build_macrocategories/1-lista_topic.csv"
    # esporta_conteggi_csv(originale, colonna, count) #crea count_indicatori che mi dice per ogni topic quanti indicatori ci sono
    clean ="data/preprocessing/coarse_wdi_dim_red/2-build_macrocategories/2-count_per_macro_topic.csv"
    distinct = "data/preprocessing/coarse_wdi_dim_red/2-build_macrocategories/3-distinct_macro_topic.csv"
    prefisso = "data/preprocessing/coarse_wdi_dim_red/2-build_macrocategories/4-macro_topic_con_topic.csv"
    topic_by_macrocat = "data/preprocessing/coarse_wdi_dim_red/3-split_dataset/1-split_by_macrocat"
    split_data = "data/preprocessing/coarse_wdi_dim_red/3-split_dataset/2-split_data"
    indicators_coverage = "data/preprocessing/coarse_wdi_dim_red/4-generate_coverage/1-all_coverage"

    # taglia_categoria(count, colonna, clean)
    # esporta_conteggi_csv(clean, colonna, distinct)
    # raggruppa_topic(only_indicatori, colonna, prefisso)

    # split_csv_per_attributo(originale, colonna, topic_by_macrocat)

    # separa_per_macrocategoria( og_data, topic_by_macrocat,
    #     colonna_codici_macrocategoria="Series Code",  # colonna nei CSV con indicatori
    #     colonna_codici_wdi="Indicator Code",               # colonna in WDICSV
    #     output_dir=split_data
    # )
    # generate_indicator_coverage(split_data, indicators_coverage)

    indicators_health = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/data/coverage_Health.csv"
    data_health = "data/preprocessing/coarse_wdi_dim_red/3-split_dataset/2-split_data/Health.csv"
    output_health1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Health_dataset_2019.csv"
    output_health2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Health_dataset_2009.csv"
    txt_health1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Health_pca_2009.txt"
    txt_health2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Health_pca_2019.txt"

    indicators_economy = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/data/coverage_Economic_Policy_&_Debt.csv"
    data_economy = "data/preprocessing/coarse_wdi_dim_red/3-split_dataset/2-split_data/Economic_Policy_&_Debt.csv"
    output_economy1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Economic_Policy_&_Debt_dataset_2019.csv"
    output_economy2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Economic_Policy_&_Debt_dataset_2009.csv"
    txt_economy1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Economic_Policy_&_Debt_pca_2009.txt"
    txt_economy2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Economic_Policy_&_Debt_pca_2019.txt"

    indicators_environment = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/data/coverage_Environment.csv"
    data_environment = "data/preprocessing/coarse_wdi_dim_red/3-split_dataset/2-split_data/Environment.csv"
    output_environment1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Environment_dataset_2019.csv"
    output_environment2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Environment_dataset_2009.csv"
    txt_environment1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Environment_pca_2009.txt"
    txt_environment2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Environment_pca_2019.txt"

    indicators_private = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/data/coverage_Private_Sector_&_Trade.csv"
    data_private = "data/preprocessing/coarse_wdi_dim_red/3-split_dataset/2-split_data/Private_Sector_&_Trade.csv"
    output_private1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Private_Sector_&_Trade_dataset_2019.csv"
    output_private2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Private_Sector_&_Trade_dataset_2009.csv"
    txt_private1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Private_Sector_&_Trade_pca_2009.txt"
    txt_private2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Private_Sector_&_Trade_pca_2019.txt"

    indicators_social = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/data/coverage_Social_Protection_&_Labor.csv"
    data_social = "data/preprocessing/coarse_wdi_dim_red/3-split_dataset/2-split_data/Social_Protection_&_Labor.csv"
    output_social1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Social_Protection_&_Labor_dataset_2019.csv"
    output_social2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/Social_Protection_&_Labor_dataset_2009.csv"
    txt_social1 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Social_Protection_&_Labor_pca_2009.txt"
    txt_social2 = "data/preprocessing/coarse_wdi_dim_red/5-attribute_selection/pca/pca_results/txt/Social_Protection_&_Labor_pca_2019.txt"


    year1 = "2019"
    year2 = "2009"
    # prepare_pca_dataset(indicators_health, data_health, output_health1, year1)
    # prepare_pca_dataset(indicators_environment, data_environment, output_environment1, year1)
    # prepare_pca_dataset(indicators_economy, data_economy, output_economy1, year1)
    # prepare_pca_dataset(indicators_private, data_private, output_private1, year1)
    # prepare_pca_dataset(indicators_social, data_social, output_social1, year1)
    #
    # prepare_pca_dataset(indicators_health, data_health, output_health2, year2)
    # prepare_pca_dataset(indicators_environment, data_environment, output_environment2, year2)
    # prepare_pca_dataset(indicators_economy, data_economy, output_economy2, year2)
    # prepare_pca_dataset(indicators_private, data_private, output_private2, year2)
    # prepare_pca_dataset(indicators_social, data_social, output_social2, year2)
    #
    # run_pca(output_health1,10, txt_health1)
    # run_pca(output_health2, 10, txt_health2)
    # run_pca(output_economy1, 10, txt_economy1)
    # run_pca(output_economy2, 10, txt_economy2)
    # run_pca(output_environment1, 10, txt_environment1)
    # run_pca(output_environment2, 10, txt_environment2)
    # run_pca(output_private1,10, txt_private1)
    # run_pca(output_private2,10, txt_private2)
    # run_pca(output_social1,10, txt_social1)
    # run_pca(output_social2,10, txt_social2)