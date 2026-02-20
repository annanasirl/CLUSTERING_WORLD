import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from clusteringalg.clique import CLIQUE

class CliqueOptimizerThread(QThread):
    finished = pyqtSignal(int, float, float)  # num_intervals, min_cluster_fraction, silhouette

    def __init__(self, X):
        super().__init__()
        self.X = X

    def run(self):
        print("Clique Optimizer")

        num_intervals_list = [20, 25, 30, 35, 40]
        min_cluster_fraction_list = [0.02, 0.05, 0.08]

        max_subspace_dim = 25
        top_dims = min(25, self.X.shape[1])

        best_score = -np.inf
        best_params = None

        for ni in num_intervals_list:
            for mcf in min_cluster_fraction_list:
                model = CLIQUE(num_intervals=ni,
                               max_subspace_dim=max_subspace_dim,
                               min_cluster_fraction=mcf,
                               top_dims=top_dims)
                model.fit(self.X)
                score = model.silhouette_score(self.X)
                print(score)
                if score > best_score:
                    best_score = score
                    best_params = (ni, mcf)

        ni_best, mcf_best = best_params
        self.finished.emit(ni_best, mcf_best, best_score)
