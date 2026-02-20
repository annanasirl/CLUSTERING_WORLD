from PyQt6.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class elbowPlotWindow(QDialog):
    def __init__(self, X, algorithm_class, max_clusters=10):
        super().__init__()
        self.setWindowTitle("Elbow Plot per k (Cosine vs Euclidean)")
        self.figure = Figure(figsize=(12, 4))  # più largo per due plot affiancati
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.X = X
        self.algorithm_class = algorithm_class
        self.max_clusters = max_clusters

        self.plot_elbow()

    def plot_elbow(self):
        distance_metrics = ['cosine', 'euclidean']

        ax_cosine = self.figure.add_subplot(1, 2, 1)
        ax_euclidean = self.figure.add_subplot(1, 2, 2)
        axes = {'cosine': ax_cosine, 'euclidean': ax_euclidean}

        for metric in distance_metrics:
            model = self.algorithm_class(
                n_clusters=1,  # valore dummy
                max_iter=100,
                random_state=42,
                distance_metric=metric
            )

            # Chiama elbow_method UNA SOLA VOLTA
            results = model.elbow_method(self.X, max_clusters=self.max_clusters)

            k_values = [r[0] for r in results]
            inertias = [r[1] for r in results]

            ax = axes[metric]
            ax.plot(k_values, inertias, marker='o')
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia")
            ax.set_title(f"Elbow Plot ({metric})")
            ax.grid(True, alpha=0.3)

        self.canvas.draw()
