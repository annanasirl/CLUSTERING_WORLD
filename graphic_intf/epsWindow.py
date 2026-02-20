from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from clusteringalg.dbscan import DBSCAN


class EpsElbowPlotWindow(QMainWindow):
    def __init__(self, X, min_samples):
        super().__init__()

        self.setWindowTitle("DBSCAN ε estimation – k-distance plot")

        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.plot_k_distance(X, min_samples)

    def plot_k_distance(self, X, min_samples):
        self.figure.clear()  # pulisce la figura

        # Crea due subplot (2 righe, 1 colonna)
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        # --- Euclidean ---
        db_euclid = DBSCAN(min_samples=min_samples, distance_metric="euclidean")
        kdist_euclid = db_euclid.elbow_method(X)

        ax1.plot(kdist_euclid)
        ax1.set_title("k-distance graph (Euclidean)")
        ax1.set_xlabel("Points sorted by k-distance")
        ax1.set_ylabel(f"k-distance (k = {min_samples})")
        ax1.grid(True)

        # --- Cosine ---
        db_cosine = DBSCAN(min_samples=min_samples, distance_metric="cosine")
        kdist_cosine = db_cosine.elbow_method(X)

        ax2.plot(kdist_cosine)
        ax2.set_title("k-distance graph (Cosine)")
        ax2.set_xlabel("Points sorted by k-distance")
        ax2.set_ylabel(f"k-distance (k = {min_samples})")
        ax2.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()