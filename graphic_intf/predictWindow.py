import os
import time
import geopandas as gpd
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import sklearn
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
)
from PyQt6.QtGui import QMovie

class predictWindow(QMainWindow):
    def __init__(self, model, algorithm: str, base_csv_2017: str):
        super().__init__()

        self.series_folder = None
        self.gif_movie = None
        self.model = model
        self.algorithm = algorithm
        self.base_csv_2017 = base_csv_2017
        self.df_2017 = pd.read_csv(base_csv_2017)
        self.resize(1200, 800)
        self.setMinimumSize(1000, 700)
        self.figure = Figure(figsize=(8, 6))

        self.setWindowTitle(f"Predict del modello ({algorithm})")

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Predict - algoritmo: {algorithm}"))

        # Bottone per eseguire predict e salvare serie
        self.run_button = QPushButton("Esegui predict anno per anno e salva CSV")
        self.run_button.clicked.connect(self.run_prediction_series)
        layout.addWidget(self.run_button)

        # Bottone per generare animazione GIF
        if algorithm in ["k-means", "clara", "hierarchical", "dbscan"]:
            self.anim_button = QPushButton("Genera animazione cluster mondiale")
            self.anim_button.clicked.connect(self.generate_cluster_gif)
            layout.addWidget(self.anim_button)

            # Label per mostrare la GIF
            self.gif_label = QLabel()
            layout.addWidget(self.gif_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Cartella di output per la serie
        self.output_folder = "../data/SERIES"
        os.makedirs(self.output_folder, exist_ok=True)

    def run_prediction_series(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        k_clusters = getattr(self.model, "n_clusters", getattr(self.model, "n_medoids", "2"))
        self.series_folder = os.path.join(self.output_folder, f"SERIES_{self.algorithm}_{k_clusters}_{timestamp}")
        os.makedirs(self.series_folder, exist_ok=True)
        print(f"Cartella creata: {self.series_folder}")

        # Salvo parametri del modello originale (per rifit)
        model_params = {}
        if hasattr(self.model, "get_params"):
            model_params = self.model.get_params()

        for year in range(2005, 2023):
            csv_path = f"../data/final_dataset/data_{year}.csv"
            df = pd.read_csv(csv_path)
            X = df.select_dtypes(include=["number"]).values

            # KMEANS / CLARA  usa predict()
            if self.algorithm.lower() in ["k-means", "clara"]:
                labels = self.model.predict(X)

            # DBSCAN / CLIQUE / HIERARCHICAL  rifit con stessi iperparametri
            elif self.algorithm.lower() in ["dbscan", "clique", "hierarchical"]:
                self.model.fit(X)
                labels = self.model.labels_

            else:
                print(f"Algoritmo non supportato: {self.algorithm}")
                return

            # Salvataggio CSV
            df_out = df.copy()
            df_out["cluster"] = labels
            out_path = os.path.join(self.series_folder, f"data_{year}_clusters.csv")
            df_out.to_csv(out_path, index=False)
            print(f"Anno {year} salvato in {out_path}")

    def generate_cluster_gif(self):
        if not hasattr(self, "series_folder") or not os.path.exists(self.series_folder):
            print("Prima esegui il predict anno per anno!")
            return

        print("Generazione animazione cluster mondiale...")

        shapefile_path = "../images/ne_50m_admin_0_countries/world_fixed.shp"
        if not os.path.exists(shapefile_path):
            print(f"Shapefile non trovato: {shapefile_path}")
            return

        # Crea cartella images dentro la serie
        images_folder = os.path.join(self.series_folder, "images")
        os.makedirs(images_folder, exist_ok=True)

        world = gpd.read_file(shapefile_path)
        gif_frames = []

        #Mappatura colori fissa
        cluster_colors = {
            0: "#1f77b4",  # blu
            1: "#ff7f0e",  # arancione
            2: "#2ca02c",  # verde
            3: "#d62728",  # rosso
            4: "#9467bd",  # lilla
            5: "#8c564b",  # marrone
        }

        gif_path = os.path.join(images_folder, f"{self.algorithm}_clusters_series.gif")

        for year in range(2005, 2023):
            csv_path = os.path.join(self.series_folder, f"data_{year}_clusters.csv")

            if not os.path.exists(csv_path):
                print(f"File mancante: {csv_path}")
                continue

            df_year = pd.read_csv(csv_path)

            #Merge robusto con shapefile
            df_plot = world.merge(
                df_year,
                left_on="ISO_A3",
                right_on="Country Code",
                how="left"
            )

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            #Disegno tutti i paesi in grigio chiaro come base
            world.plot(ax=ax, color="lightgrey", edgecolor="white")

            # Disegno cluster con colori fissi
            for c, color in cluster_colors.items():
                subset = df_plot[df_plot["cluster"] == c]
                if not subset.empty:
                    subset.plot(ax=ax, color=color)

            ax.set_title(f"{self.algorithm.upper()} - Cluster {year}")
            ax.axis("off")

            #Salva PNG definitivo
            frame_path = os.path.join(images_folder, f"cluster_{year}.png")
            fig.savefig(frame_path, bbox_inches="tight")
            plt.close(fig)

            gif_frames.append(frame_path)

        #Scrivi GIF
        with imageio.get_writer(gif_path, mode="I", duration=1.0) as writer:
            for frame in gif_frames:
                img = imageio.imread(frame)
                writer.append_data(img)

        print(f"GIF generata in: {gif_path}")

        # Mostra GIF nella finestra
        self.gif_movie = QMovie(gif_path)
        self.gif_label.setMovie(self.gif_movie)
        self.gif_movie.start()
