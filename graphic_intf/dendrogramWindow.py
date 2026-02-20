from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class DendrogramWindow(QDialog):
    def __init__(self, X, method='average', metric='cosine', labels=None, min_cluster_size_ratio=0.05):
        """
        X: dati numerici
        method: linkage ('single', 'complete', 'average')
        metric: 'euclidean' o 'cosine'
        labels: nomi delle foglie
        min_cluster_size_ratio: dimensione minima cluster come frazione del totale
        """
        super().__init__()
        self.setWindowTitle("Dendrogramma Hierarchical Clustering")
        self.setMinimumSize(800, 700)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Figura
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Label per k ottimale
        self.opt_k_label = QLabel("")
        self.opt_k_label.setWordWrap(True)  # testo su più righe
        layout.addWidget(self.opt_k_label)

        # Normalizzazione se cosine
        if metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

        n_samples = X.shape[0]

        # Linkage
        Z = linkage(X, method=method, metric=metric)

        # Disegno dendrogramma
        ax = self.figure.add_subplot(111)
        dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90)
        ax.set_title("Dendrogram")
        ax.set_ylabel("Distanza")
        self.canvas.draw()

        # --- Gap massimo tra fusioni ---
        distances = Z[:, 2]          # distanze di fusione
        diffs = np.diff(distances)   # incremento tra fusioni
        max_jump_idx = np.argmax(diffs)
        gap_k = n_samples - (max_jump_idx + 1)

        # --- Applica vincolo sulla dimensione minima dei cluster ---
        min_size = int(n_samples * min_cluster_size_ratio)
        candidate_k = gap_k
        while True:
            cluster_labels = fcluster(Z, t=candidate_k, criterion='maxclust')
            sizes = [np.sum(cluster_labels == i) for i in np.unique(cluster_labels)]
            if all(s >= min_size for s in sizes):
                break
            candidate_k -= 1
            if candidate_k <= 1:
                candidate_k = 1
                break

        self.opt_k_label.setText(f"Optimal number of clusters (gap + min size {min_size}): {candidate_k}")
