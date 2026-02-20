import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4,
                 distance_metric='euclidean', random_state=None):

        self.n_clusters = n_clusters                #k cluster
        self.max_iter = max_iter                    #quante iterazioni
        self.tol = tol                              #soglia di convergenza
        self.distance_metric = distance_metric      #distance metric scelta
        self.random_state = random_state            #x random sampling
        self.centroids = None                       #conterrà i centroidi
        self.labels_ = None                         #conterrà le label dei pt

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
        if self.distance_metric == 'euclidean':
            return np.array([self._euclidean(x, c) for c in centers])
        elif self.distance_metric == 'cosine':
            return np.array([self._cosine(x, c) for c in centers])
        else:
            raise ValueError("Distanza non supportata")


    def fit(self, X):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=float)

        if self.distance_metric == "cosine":
            X = self._l2_normalize(X)

        n_samples, n_features = X.shape

        #inizializza k centroidi casuali
        init_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[init_idx].copy()

        #per ogni iterazione
        for iteration in range(self.max_iter):
            #assegno un label a ogni pt in base a centroid + vicino
            labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                labels[i] = np.argmin(self._distance(X[i], self.centroids))

            #per ogni cluster, se ci sono i pt, calcolo nuovo centroid
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                points = X[labels == k]
                if len(points) > 0:
                    new_centroids[k] = np.mean(points, axis=0)
                    if self.distance_metric == 'cosine':
                        norm = np.linalg.norm(new_centroids[k])
                        if norm > 0:
                            new_centroids[k] /= norm
                else:
                    new_centroids[k] = X[np.random.randint(0, n_samples)]
                    #(se cluster vuoto prendo un altro centroide a random)

            #controllo se convergenza in base a tolleranza
            shift = np.sum((new_centroids - self.centroids) ** 2)
            if shift < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = labels

    # ---------------------------
    # Funzione predict
    # ---------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.distance_metric == 'cosine':
            X = self._l2_normalize(X)
        return np.array([np.argmin(self._distance(x, self.centroids)) for x in X])

    def elbow_method(self, X, max_clusters=10):
        """
        Calcola l'inertia (somma delle distanze quadratiche) per un range di cluster
        per identificare il 'gomito' (elbow).

        Ritorna:
            list of tuples: [(n_clusters, inertia), ...]
        """
        X = np.asarray(X, dtype=float)
        if self.distance_metric == 'cosine':
            X = self._l2_normalize(X)

        inertias = []
        for k in range(1, max_clusters + 1):
            km = KMeans(n_clusters=k, max_iter=self.max_iter,
                        distance_metric=self.distance_metric,
                        random_state=self.random_state)
            km.fit(X)
            # inertia = somma dei quadrati delle distanze dei punti dal centroide
            inertia = 0
            for i in range(k):
                points = X[km.labels_ == i]
                inertia += np.sum([self._distance(p, [km.centroids[i]])[0] ** 2 for p in points])
            inertias.append((k, inertia))
        return inertias

    def silhouette_score(self, X):
        if self.labels_ is None:
            raise ValueError("Devi chiamare fit() prima di silhouette_score().")

        if self.n_clusters < 2:
            raise ValueError("Silhouette score richiede almeno 2 cluster.")

        X = np.asarray(X, dtype=float)

        if self.distance_metric == 'cosine':
            X = self._l2_normalize(X)

        labels = self.labels_
        unique_labels = np.unique(labels)
        n_samples = X.shape[0]
        score = 0.0

        for i in range(n_samples):
            own_label = labels[i]
            own_cluster = X[labels == own_label]

            # a(i) distanza media intra-cluster
            if len(own_cluster) > 1:
                a_i = np.mean([
                    self._distance(X[i], [p])[0]
                    for p in own_cluster
                    if not np.array_equal(p, X[i])
                ])
            else:
                a_i = 0.0

            # b(i) minima distanza media verso altri cluster nn vuoti
            b_values = []
            for k in unique_labels:
                if k == own_label:
                    continue
                cluster_k = X[labels == k]
                if len(cluster_k) == 0:
                    continue
                b_values.append(
                    np.mean([self._distance(X[i], [p])[0] for p in cluster_k])
                )

            b_i = min(b_values) if b_values else 0.0

            denom = max(a_i, b_i)
            s_i = (b_i - a_i) / denom if denom > 0 else 0.0
            score += s_i

        return score / n_samples

    def silhouette_samples(self, X):
        if self.labels_ is None:
            raise ValueError("Devi chiamare fit() prima di silhouette_samples().")

        if self.n_clusters < 2:
            raise ValueError("Silhouette richiede almeno 2 cluster.")

        X = np.asarray(X, dtype=float)

        if self.distance_metric == 'cosine':
            X = self._l2_normalize(X)

        labels = self.labels_
        unique_labels = np.unique(labels)
        n_samples = X.shape[0]
        s = np.zeros(n_samples)

        for i in range(n_samples):
            own_label = labels[i]
            own_cluster = X[labels == own_label]

            # a(i)
            if len(own_cluster) > 1:
                a_i = np.mean([
                    self._distance(X[i], [p])[0]
                    for p in own_cluster
                    if not np.array_equal(p, X[i])
                ])
            else:
                a_i = 0.0

            # b(i)
            b_values = []
            for k in unique_labels:
                if k == own_label:
                    continue
                cluster_k = X[labels == k]
                if len(cluster_k) == 0:
                    continue
                b_values.append(
                    np.mean([self._distance(X[i], [p])[0] for p in cluster_k])
                )

            b_i = min(b_values) if b_values else 0.0

            denom = max(a_i, b_i)
            s[i] = (b_i - a_i) / denom if denom > 0 else 0.0

        return s

