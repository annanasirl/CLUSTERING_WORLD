from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import pycountry

class worldMapWindow(QMainWindow):
    def __init__(self, countries, labels, year, algorithm, cluster_colors):
        super().__init__()
        self.setWindowTitle(f"Mappa Cluster {year} - {algorithm}")
        self.setMinimumSize(800, 600)

        self.countries = countries
        self.labels = labels
        self.year = year
        self.algorithm = algorithm
        self.cluster_colors = cluster_colors

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.plot_map()

    def plot_map(self):
        ax = self.figure.add_subplot(111)

        # --- CARICA GEO DATAFRAME LOCALE ---
        shapefile_path = "../images/ne_50m_admin_0_countries/world_fixed.shp"
        world = gpd.read_file(shapefile_path)

        # --- FUNZIONE PER CONVERTIRE ISO2 / NOME IN ISO3 ---
        def to_iso3(code_or_name):
            if not isinstance(code_or_name, str):
                return None
            code_or_name = code_or_name.strip().upper()
            # prova ISO2
            try:
                return pycountry.countries.get(alpha_2=code_or_name).alpha_3
            except:
                pass
            # prova nome
            try:
                return pycountry.countries.lookup(code_or_name).alpha_3
            except:
                return None

        # dataframe dei cluster con ISO3
        df_map = pd.DataFrame({
            "cluster": self.labels,
            "Country Code": [to_iso3(c) for c in self.countries]
        })

        # --- MERGE COMPLETO ---
        merged = world.merge(df_map, left_on="ISO_A3", right_on="Country Code", how="left")

        # --- CHECK PAESI MANCANTI ---
        missing = df_map[~df_map['Country Code'].isin(world['ISO_A3'])]
        if not missing.empty:
            print("Attenzione! Paesi mancanti nello shapefile:", missing['Country Code'].tolist())

        # --- COLORI DEI CLUSTER ---
        merged['color'] = merged['cluster'].map(self.cluster_colors)
        # riempi i NaN con grigio
        merged['color'] = merged['color'].fillna('lightgrey')


        # --- PLOT ---
        merged.plot(ax=ax, color=merged['color'], edgecolor='black')
        ax.set_title(f"Mappa Cluster {self.year} - {self.algorithm}", fontsize=14)
        ax.axis('off')

        self.canvas.draw()
