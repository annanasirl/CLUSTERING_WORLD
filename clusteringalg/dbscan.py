import numpy as np
from collections import deque

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, distance_metric='euclidean'):
        """
        distance_metric: 'euclidean' o 'cosine'
        """
        self.eps = eps                              #raggio di densità epsilon
        self.min_samples = min_samples              #numero min di pt da trovare in epsilon per considerare pt core point
        self.distance_metric = distance_metric      #che distance metric usiamo
        self.labels_ = None                         #clsuter labels
        self._X_normalized = None                   #X normalizzata se cosine

    # metodo per calcolare normalizzazione L2 (normalizza dataset riga x riga x cosine)

    def _l2_normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

    # distanza euclidea tra 2 pt

    def _euclidean(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    # distanza cosine tra 2 pt normalizzati

    def _cosine(self, x, y):
        return 1 - np.dot(x, y)

    # unica funzione di distanza che la calcola in base alla metrica inserita

    def _distance(self, x, centers):
        centers = np.atleast_2d(centers)
        if self.distance_metric == 'euclidean':
            dists = np.sqrt(np.sum((centers - x) ** 2, axis=1))
        elif self.distance_metric == 'cosine':
            dists = 1 - np.dot(centers, x)
        else:
            raise ValueError("Distanza non supportata")

        # se centers era un singolo punto, ritorna scalare
        if dists.size == 1:
            return dists[0]
        return dists

    #trova tutti i pt che stanno nel raggio di eps da punto idx
    def _region_query(self, X, idx):
        return [i for i in range(len(X)) if self._distance(X[idx], X[i]) <= self.eps]


    def fit(self, X):
        X = np.asarray(X, dtype=float)

        #normalizzazione L2 se cosine
        if self.distance_metric == 'cosine':
            self._X_normalized = self._l2_normalize(X)
        else:
            self._X_normalized = X

        #conto samples, metto tutte labels a -1, segno che nn ho ancora visitato punti
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        #tengo traccia di cluster id partendo da 0
        cluster_id = 0

        #per ogni punto
        for i in range(n_samples):
            #se già visitato passo al prossimo
            if visited[i]:
                continue

            #lo visito
            visited[i] = True
            #cerco i suoi "vicini"
            neighbors = self._region_query(self._X_normalized, i)

            #se i suoi vicini sono più di minsamples, espando cluster.
            if len(neighbors) < self.min_samples:
                continue  # resta rumore
            else:
                cluster_id += 1
                #pt i è in cluster cluster_id
                labels[i] = cluster_id
                #double ended queuq dei vicini
                queue = deque(neighbors)

                #per ogni vicino
                while queue:
                    j = queue.popleft()
                    if not visited[j]:
                        visited[j] = True
                        #prendo il suo vicinato
                        j_neighbors = self._region_query(self._X_normalized, j)
                        #se anche lui è core metto i suoi vicini nella coda dei vicini
                        if len(j_neighbors) >= self.min_samples:
                            queue.extend(j_neighbors)
                    if labels[j] == -1:
                        labels[j] = cluster_id
                        #se non è ancora assegnato a un cluster lo assegno al cluster corrente

        self.labels_ = labels
        return self


    def elbow_method(self, X, k=None):
        X = np.asarray(X, dtype=float)
        if self.distance_metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

        n_samples = X.shape[0]
        if k is None:
            k = self.min_samples

        k_distances = []
        for i in range(n_samples):
            distances = [self._distance(X[i], X[j]) for j in range(n_samples) if i != j]
            distances.sort()
            k_distances.append(distances[k-1])

        k_distances = np.array(k_distances)
        k_distances.sort()
        return k_distances


    def silhouette_samples(self, X):
        if self.labels_ is None:
            raise ValueError("Devi chiamare fit() prima.")

        X = np.asarray(X, dtype=float)
        if self.distance_metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

        labels = self.labels_
        n_samples = len(X)
        unique_labels = [k for k in np.unique(labels) if k != -1]
        s = np.zeros(n_samples)

        for i in range(n_samples):
            if labels[i] == -1:
                s[i] = 0
                continue

            own_cluster = X[labels == labels[i]]
            if len(own_cluster) > 1:
                a_i = np.mean([self._distance(X[i], p) for p in own_cluster if not np.array_equal(p, X[i])])
            else:
                s[i] = 0
                continue

            b_i = np.inf
            for k in unique_labels:
                if k == labels[i]:
                    continue
                cluster_k = X[labels == k]
                dist = np.mean([self._distance(X[i], p) for p in cluster_k])
                if dist < b_i:
                    b_i = dist

            s[i] = 0 if max(a_i, b_i) == 0 else (b_i - a_i) / max(a_i, b_i)

        return s

    def silhouette_score(self, X):
        s = self.silhouette_samples(X)
        valid = self.labels_ != -1
        if np.sum(valid) == 0:
            return 0
        return np.mean(s[valid])
