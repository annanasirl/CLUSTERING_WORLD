from PyQt6.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class SilhouettePlotWindow(QDialog):
    def __init__(self, X, labels, model, cluster_colors):
        super().__init__()
        self.setWindowTitle("Silhouette Plot")

        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.cluster_colors = cluster_colors

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.X = X
        self.labels = labels
        self.model = model  # oggetto che espone silhouette_samples / silhouette_score

        self.plot_silhouette()

    def plot_silhouette(self):
        ax = self.figure.add_subplot(111)

        silhouette_vals = self.model.silhouette_samples(self.X)
        y_lower = 10

        for cluster in np.unique(self.labels):
            cluster_silhouette_vals = silhouette_vals[self.labels == cluster]
            cluster_silhouette_vals.sort()

            y_upper = y_lower + len(cluster_silhouette_vals)

            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_vals,
                alpha=0.7,
                color = self.cluster_colors[cluster],
                label=f"Cluster {cluster}"
            )

            y_lower = y_upper + 10

        avg_score = self.model.silhouette_score(self.X)
        ax.axvline(avg_score, linestyle="--")
        ax.set_title("Silhouette Plot")
        ax.set_xlabel(f"Silhouette coefficient - avg {avg_score}")
        ax.set_ylabel("Cluster")

        ax.legend()
        self.canvas.draw()
