from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from clusteringalg.clique import CLIQUE
from clusteringalg.kmeans import KMeans
from clusteringalg.clara import CLARA
from clusteringalg.dbscan import DBSCAN
from clusteringalg.hierarchical import Hierarchical
from statsWindow import ClusterStatsWindow
from elbowWindow import elbowPlotWindow
from silhouetteWindow import SilhouettePlotWindow
from epsWindow import EpsElbowPlotWindow
from worldMapWindow import worldMapWindow
from predictWindow import predictWindow
from dendrogramWindow import DendrogramWindow
from cliqueOpt import CliqueOptimizerThread
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QToolTip
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

class ClusterWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.clique_thread = None
        self.clique_result_window = None
        self.cluster_colors = None
        self.countries = None
        self.elb_window = None
        self.stats_window = None
        self.silhouette_window = None
        self.cl_path = None
        self.pred_model = None
        self.setWindowTitle("CCISED - clustering")

        cl_layout = QVBoxLayout()
        cl_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        title = QLabel("CCISED")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        cl_layout.addWidget(title)

        description = QLabel("Interfaccia per selezionare algoritmo di clustering e iperparametri")
        description.setAlignment(Qt.AlignmentFlag.AlignLeft)
        cl_layout.addWidget(description)

        # Dropdown per l'anno
        self.year_label = QLabel("Scegli anno: ")
        self.year = QComboBox()
        self.year.addItems([str(y) for y in range(2005, 2023)])
        cl_layout.addWidget(self.year_label)
        cl_layout.addWidget(self.year)

        # Dropdown per l'algoritmo
        self.algorithm_label = QLabel("Scelta Algoritmo: ")
        self.algorithm = QComboBox()
        self.algorithm.addItem("")
        self.algorithm.addItems(['k-means', 'clara', 'clique', 'hierarchical', 'dbscan'])
        cl_layout.addWidget(self.algorithm_label)
        cl_layout.addWidget(self.algorithm)
        self.algorithm.currentTextChanged.connect(self.update_algorithm_options)

        # Opzioni dinamiche
        # Per k-means: numero di cluster

        self.find_k_button = QPushButton("Individua numero K di cluster")
        self.find_k_button.setFixedSize(QSize(400, 40))
        cl_layout.addWidget(self.find_k_button)
        self.find_k_button.clicked.connect(self.show_elbow_plot)

        self.k_label = QLabel("Numero di cluster:")
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(2)
        self.k_spinbox.setMaximum(20)
        cl_layout.addWidget(self.k_label)
        cl_layout.addWidget(self.k_spinbox)
        self.k_label.hide()
        self.k_spinbox.hide()
        self.find_k_button.hide()

        # Per DBSCAN: eps e min_samples

        self.min_samples_label = QLabel("Min Samples:")
        self.min_samples_input = QSpinBox()
        self.min_samples_input.setMinimum(1)
        self.min_samples_input.setMaximum(100)

        self.find_eps_button = QPushButton("Individua il numero ideale per epsilon: ")
        self.find_eps_button.setFixedSize(QSize(400, 40))
        cl_layout.addWidget(self.find_eps_button)
        self.find_eps_button.clicked.connect(self.show_eps_plot)

        self.eps_label = QLabel("Epsilon:")
        self.eps_input = QDoubleSpinBox()
        self.eps_input.setDecimals(2)
        self.eps_input.setSingleStep(0.1)
        self.eps_input.setRange(0.01, 10.0)

        cl_layout.addWidget(self.eps_label)
        cl_layout.addWidget(self.eps_input)
        cl_layout.addWidget(self.min_samples_label)
        cl_layout.addWidget(self.min_samples_input)

        self.find_eps_button.hide()
        self.eps_label.hide()
        self.eps_input.hide()
        self.min_samples_label.hide()
        self.min_samples_input.hide()

        # CLIQUE

        self.optimize_clique_button = QPushButton("Ottimizza parametri CLIQUE")
        self.optimize_clique_button.setFixedSize(QSize(400, 40))
        self.optimize_clique_button.clicked.connect(self.optimize_clique_params)
        cl_layout.addWidget(self.optimize_clique_button)
        self.optimize_clique_button.hide()
        
        self.num_intervals_label = QLabel("Numero di intervalli (CLIQUE):")
        self.num_intervals_input = QSpinBox()
        self.num_intervals_input.setRange(2, 40)
        self.num_intervals_input.setValue(2)

        self.max_dim_label = QLabel("Maximum subspace dimensions (CLIQUE):")
        self.max_dim_input = QSpinBox()
        self.max_dim_input.setRange(1, 25)
        self.max_dim_input.setSingleStep(1)
        self.max_dim_input.setValue(1)

        self.mcf_label = QLabel("Min. cluster fraction:")
        self.mcf_input = QDoubleSpinBox()
        self.mcf_input.setDecimals(2)
        self.mcf_input.setSingleStep(0.01)
        self.mcf_input.setRange(0.01, 1.00)

        self.top_dim_label = QLabel("Top dimensions (CLIQUE):")
        self.top_dim_input = QSpinBox()
        self.top_dim_input.setRange(1, 25)
        self.top_dim_input.setSingleStep(1)
        self.top_dim_input.setValue(1)

        cl_layout.addWidget(self.num_intervals_label)
        cl_layout.addWidget(self.num_intervals_input)
        cl_layout.addWidget(self.max_dim_label)
        cl_layout.addWidget(self.max_dim_input)
        cl_layout.addWidget(self.mcf_label)
        cl_layout.addWidget(self.mcf_input)
        cl_layout.addWidget(self.top_dim_label)
        cl_layout.addWidget(self.top_dim_input)

        self.num_intervals_label.hide()
        self.num_intervals_input.hide()
        self.max_dim_label.hide()
        self.max_dim_input.hide()
        self.mcf_label.hide()
        self.mcf_input.hide()
        self.top_dim_input.hide()
        self.top_dim_label.hide()

        # HIERARCHICAL
        self.linkage_label = QLabel("Linkage: ")
        self.linkage_data = QComboBox()
        self.linkage_data.addItems(['single', 'complete', 'average'])
        cl_layout.addWidget(self.linkage_label)
        cl_layout.addWidget(self.linkage_data)
        self.linkage_label.hide()
        self.linkage_data.hide()

        self.distance_label = QLabel("Distance measure: ")
        self.distance_data = QComboBox()
        self.distance_data.addItems(['euclidean', 'cosine'])
        cl_layout.addWidget(self.distance_label)
        cl_layout.addWidget(self.distance_data)
        self.distance_label.hide()
        self.distance_data.hide()

        # Bottone per lanciare il clustering
        self.button1 = QPushButton("Clustering")
        self.button1.setFixedSize(QSize(400, 50))
        cl_layout.addWidget(self.button1)

        # Connetti il bottone alla funzione start_clustering
        self.button1.clicked.connect(self.start_clustering)

        self.button2 = QPushButton("Predict")
        self.button2.setFixedSize(QSize(400, 50))
        cl_layout.addWidget(self.button2)
        self.button2.hide()

        self.button2.clicked.connect(self.start_predicting)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(500)
        self.toolbar = NavigationToolbar(self.canvas, self)
        cl_layout.addWidget(self.toolbar)
        cl_layout.addWidget(self.canvas, stretch=1)

        cl_layout.setStretchFactor(self.canvas, 5)

        w = QWidget()
        w.setLayout(cl_layout)
        self.setCentralWidget(w)

    def update_algorithm_options(self, algorithm_name):
        # Nascondi tutto prima
        for widget in [self.k_label, self.k_spinbox,
                       self.eps_label, self.eps_input,
                       self.min_samples_label, self.min_samples_input,
                       self.num_intervals_label, self.num_intervals_input,
                       self.max_dim_label, self.max_dim_input,
                       self.linkage_label, self.linkage_data,
                       self.distance_label, self.distance_data,
                       self.find_eps_button, self.find_k_button,
                       self.top_dim_label, self.top_dim_input,
                       self.mcf_label, self.mcf_input, self.button2, self.optimize_clique_button]:
            widget.hide()

        if algorithm_name in ['k-means', 'clara']:
            self.find_k_button.show()
            self.k_label.show()
            self.k_spinbox.show()
            self.distance_label.show()
            self.distance_data.show()
        elif algorithm_name == 'hierarchical':
            self.k_label.show()
            self.k_spinbox.show()
            self.linkage_label.show()
            self.linkage_data.show()
            self.distance_label.show()
            self.distance_data.show()
        elif algorithm_name == 'dbscan':
            self.min_samples_label.show()
            self.min_samples_input.show()
            self.find_eps_button.show()
            self.eps_label.show()
            self.eps_input.show()
            self.distance_label.show()
            self.distance_data.show()
        elif algorithm_name == 'clique':
            self.optimize_clique_button.show()
            self.num_intervals_label.show()
            self.num_intervals_input.show()
            self.max_dim_label.show()
            self.max_dim_input.show()
            self.mcf_label.show()
            self.mcf_input.show()
            self.top_dim_input.show()
            self.top_dim_label.show()

    def show_elbow_plot(self):
        selected_algorithm = self.algorithm.currentText()
        if selected_algorithm not in ['k-means', 'clara']:
            return

        csv_path = f"../data/final_dataset/data_{self.year.currentText()}.csv"
        X = np.nan_to_num(pd.read_csv(csv_path).iloc[:, 2:].values)

        # Seleziona la classe corretta
        alg_class = KMeans if selected_algorithm == 'k-means' else CLARA

        self.elb_window = elbowPlotWindow(X, alg_class)
        self.elb_window.show()

    def show_eps_plot(self):
        selected_algorithm = self.algorithm.currentText()
        if selected_algorithm != 'dbscan':
            return

        csv_path = f"../data/final_dataset/data_{self.year.currentText()}.csv"
        X = np.nan_to_num(pd.read_csv(csv_path).iloc[:, 2:].values)

        min_samples = self.min_samples_input.value()

        self.elb_window = EpsElbowPlotWindow(X, min_samples)
        self.elb_window.show()

    def start_clustering(self):
        selected_algorithm = self.algorithm.currentText()
        if selected_algorithm == "":
            print("Seleziona un algoritmo prima di partire!")
            return
        selected_year = self.year.currentText()
        k = self.k_spinbox.value()
        eps = self.eps_input.value()
        min_samples = self.min_samples_input.value()
        num_intervals = self.num_intervals_input.value()
        dim = self.max_dim_input.value()
        linkage = self.linkage_data.currentText()
        distance = self.distance_data.currentText()
        top_dim = self.top_dim_input.value()
        mcf = self.mcf_input.value()

        print(f"Avvio clustering per l'anno {selected_year} usando {selected_algorithm}")

        self.run_clustering(selected_year, selected_algorithm, k, eps, min_samples, num_intervals, dim, linkage, distance, top_dim, mcf)


    def run_clustering(self, year, algorithm, k, eps, min_samples, num_intervals, dim, linkage, distance, top_dim, mcf):


        csv_path = f"../data/final_dataset/data_{year}.csv"
        df = pd.read_csv(csv_path)

        X = df.iloc[:, 2:].values
        self.countries = df['Country Code'].values

        if algorithm == 'k-means':
            model = KMeans(n_clusters=k, max_iter=100, random_state=42, distance_metric=distance)
        elif algorithm == 'clara':
            model = CLARA(n_clusters=k, max_iter=100, random_state=42, distance_metric=distance)
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples, distance_metric=distance)
        elif algorithm == 'hierarchical':
            model = Hierarchical(n_clusters=k, linkage=linkage, distance=distance)
        elif algorithm == 'clique':
            model = CLIQUE(num_intervals=num_intervals, max_subspace_dim=dim, min_cluster_fraction=mcf, top_dims=top_dim)
        else:
            return

        model.fit(X)
        labels = model.labels_

        self.stats_window = ClusterStatsWindow(df, labels, "std")
        self.stats_window.show()

        if algorithm == 'hierarchical':
            # Mostriamo dendrogramma
            self.dendro_window = DendrogramWindow(X, method=linkage, metric=distance, labels=self.countries)
            self.dendro_window.show()

        # === SALVATAGGIO CSV CLUSTERING ===
        df_out = df.copy()
        df_out["cluster"] = labels

        if algorithm == 'k-means':
            self.cl_path = f"../data/CLUSTERING/KMEANS/data_{year}_clusters_alg{algorithm}_{k}.csv"
            df_out.to_csv(self.cl_path, index=False)
        elif algorithm == 'clara':
            self.cl_path = f"../data/CLUSTERING/CLARA/data_{year}_clusters_alg{algorithm}_{k}.csv"
            df_out.to_csv(self.cl_path, index=False)
        elif algorithm == 'dbscan':
            self.cl_path = f"../data/CLUSTERING/DBSCAN/data_{year}_clusters_alg{algorithm}_{eps}_{min_samples}.csv"
            df_out.to_csv(self.cl_path, index=False)
        elif algorithm == 'clique':
            self.cl_path = f"../data/CLUSTERING/CLIQUE/data_{year}_clusters_alg{algorithm}_{num_intervals}_{dim}.csv"
            df_out.to_csv(self.cl_path, index=False)
        elif algorithm == 'hierarchical':
            self.cl_path = f"../data/CLUSTERING/HIERARCHICAL/data_{year}_clusters_alg{algorithm}_{k}_{linkage}.csv"
            df_out.to_csv(self.cl_path, index=False)

        print(f"CSV clustering salvato.")


        self.pred_model = model
        self.button2.show()

        X_vis = np.nan_to_num(X[:, :-1].astype(float))
        X_2d = PCA(n_components=2).fit_transform(X_vis)

        self.figure.clear()


        ax1 = self.figure.add_subplot(111)

        # --- SCATTER PCA ---
        self.scatter_points = []
        self.point_labels = []

        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab10")

        self.cluster_colors = {
            cluster_id: cmap(i % 10)
            for i, cluster_id in enumerate(sorted(unique_labels))
        }

        for cluster_id in unique_labels:
            points = X_2d[labels == cluster_id]
            label_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Outlier"
            sc = ax1.scatter(points[:, 0], points[:, 1], label=label_name, color=self.cluster_colors[cluster_id],
                             picker=True)
            self.scatter_points.append(sc)
            self.point_labels.append(self.countries[labels == cluster_id])

        ax1.set_title(f"Clustering {year} - {algorithm}")
        ax1.set_xlabel("PCA 1")
        ax1.set_ylabel("PCA 2")
        ax1.legend()

        # --- MAPPA MONDIALE ---
        self.map_window = worldMapWindow(self.countries, labels, year, algorithm, self.cluster_colors)
        self.map_window.show()

        X_numeric = df.select_dtypes(include=['number']).values
        self.silhouette_window = SilhouettePlotWindow(X_numeric, labels, model, self.cluster_colors)
        self.silhouette_window.show()

        self.canvas.draw()

        # --- TOOLTIP PCA ---
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def on_hover(self, event):
        """Mostra tooltip con il nome del paese al passaggio del mouse"""
        if event.inaxes:
            found = False
            for sc, labels in zip(self.scatter_points, self.point_labels):
                cont, ind = sc.contains(event)
                if cont:
                    country_name = labels[ind["ind"][0]]
                    QToolTip.showText(QCursor.pos(), country_name, self)
                    found = True
                    break
            if not found:
                QToolTip.hideText()

    def start_predicting(self):
        if self.cl_path is None:
            print("Nessun clustering disponibile.")
            return

        algorithm = self.algorithm.currentText()

        self.predict_window = predictWindow(
            model = self.pred_model,
            algorithm=algorithm,
            base_csv_2017=self.cl_path
        )
        self.predict_window.show()

    def optimize_clique_params(self):
        selected_year = self.year.currentText()
        csv_path = f"../data/final_dataset/data_{selected_year}.csv"
        df = pd.read_csv(csv_path)
        X = np.nan_to_num(df.iloc[:, 2:].values)

        # Creo thread
        self.clique_thread = CliqueOptimizerThread(X)
        self.clique_thread.finished.connect(self.show_clique_results)
        self.clique_thread.start()

    def show_clique_results(self, ni_best, mcf_best, best_score):
        result_window = QWidget()
        result_window.setWindowTitle("Parametri ottimali CLIQUE")
        layout = QVBoxLayout()

        layout.addWidget(QLabel(f"Numero intervalli ottimale: {ni_best}"))
        layout.addWidget(QLabel(f"Min cluster fraction ottimale: {mcf_best:.2f}"))
        layout.addWidget(QLabel(f"Silhouette score: {best_score:.3f}"))

        result_window.setLayout(layout)
        result_window.setFixedSize(300, 150)
        result_window.show()

        self.clique_result_window = result_window