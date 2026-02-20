import heapq

import numpy as np

class Hierarchical:

    def __init__(self, n_clusters=3, linkage="average", distance="cosine"):
        self.n_clusters = n_clusters        #n clusters
        self.linkage = linkage              #tipo linkage
        self.distance_metric = distance     #distance metric da usare
        self.labels_ = None                 #alla fine contiene etichette cluster

        if self.linkage not in ["average", "single", "complete"]:
            raise ValueError("linkage deve essere 'average', 'single' o 'complete'")
        if self.distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance deve essere 'cosine' o 'euclidean'")


    #metodo per calcolare normalizzazione L2 (normalizza dataset riga x riga x cosine)

    def _l2_normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return X / norms

    #distanza euclidea tra 2 pt

    def _euclidean(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    # distanza cosine tra 2 pt normalizzati

    def _cosine(self, x, y):
        return 1 - np.dot(x, y)

    # unica funzione di distanza che la calcola in base alla metrica inserita
    def _distance(self, x, y):
        if self.distance_metric == "euclidean":
            return self._euclidean(x, y)
        elif self.distance_metric == "cosine":
            return self._cosine(x, y)
        else:
            raise ValueError(f"Distanza {self.distance_metric} non supportata")

    # metodo x costruire matrice delle distanze. n numero di punti
    # prima crea matrice quadrata inizializzata a 0
    #per ogni punto calcola il triangolo superiore e inserisce distanza tra ptt i e j
    #alla fine crea matrice simmetrica

    def _compute_distance_matrix(self, X):
        n = X.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self._distance(X[i], X[j])
                D[i, j] = d
                D[j, i] = d
        return D

    # metodo per calcolare distanza tra due cluster in base alla metrica di linkage
    #prende in input due insiemi di indici che chiamiamo cluster a e b
    #in distances mettiamo le distanze tra ttt i pt di a (righe)e di b(colonne)
    #linkage average ritorna la media, single il min, complete il max
    def _cluster_distance(self, cluster_a, cluster_b, D):
        distances = [D[i, j] for i in cluster_a for j in cluster_b]
        if self.linkage == "average":
            return np.mean(distances)
        elif self.linkage == "single":
            return np.min(distances)
        elif self.linkage == "complete":
            return np.max(distances)
        else:
            raise ValueError(f"Linkage {self.linkage} non supportata")

    # funzione fit
    def fit(self, X):
        X = np.asarray(X, dtype=float)

        # all'inizio, se usiamo distanza cosine, normaliziamo dataset
        #distanza cosine è più adatta per il mio dataset ma le ho implementate tutte e due
        if self.distance_metric == "cosine":
            X = self._l2_normalize(X)

        #calcolo numero di punti e creo la distance matrix (!!)
        n_samples = X.shape[0]
        D = self._compute_distance_matrix(X)

        #inizialmente ogni pt è un cluster separato
        clusters = {i: {i} for i in range(n_samples)}
        #mantiene le dimensioni dei cluster
        cluster_sizes = {i: 1 for i in range(n_samples)}
        #quali sono i cluster attivi
        active = set(range(n_samples))

        #creiamo un min heap vuoto che ci serve per cercare dist minima in modo più ottimizzato
        heap = []

        # inserisco tutte le distanze tra cluster (ora tra punti!) iniziali nell'heap
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = D[i, j]
                heapq.heappush(heap, (dist, i, j))

        #id del nuovo cluster
        next_cluster_id = n_samples

        # finchè non rimangono esattamente k clusters
        while len(active) > self.n_clusters:

            # estraggo la distanza minima tra due cluster che non sono stati scartati
            while True:
                dist, i, j = heapq.heappop(heap)
                if i in active and j in active:
                    break

            #crea nuovo cluster unendo i due cluster scelti, lo salvo e aggiorno la dim
            new_cluster = clusters[i].union(clusters[j])
            clusters[next_cluster_id] = new_cluster
            cluster_sizes[next_cluster_id] = (
                    cluster_sizes[i] + cluster_sizes[j]
            )

            #disattivo i vecchi cluster così se pescati da heap vengono scartati, aggiungo nuovo
            active.remove(i)
            active.remove(j)
            active.add(next_cluster_id)

            #scorro tutti i cluster attivi e calcolo la distanza solo col nuovo cluster
            #poi la aggiungo nell heap
            for k in active:
                if k == next_cluster_id:
                    continue

                dist_new = self._cluster_distance(
                    clusters[next_cluster_id], clusters[k], D
                )

                heapq.heappush(heap, (dist_new, next_cluster_id, k))
            #incremento id del nuovo cluster
            next_cluster_id += 1

        #a questo punto ho esattamente k clsuters
        #in labels metto le etichette per ora tutte 0
        labels = np.zeros(n_samples, dtype=int)

        #scorro cluster attivi e assegno a ogni pt etichetta relativa a cluster attivo in cui si trova
        for label, cluster_id in enumerate(active):
            for idx in clusters[cluster_id]:
                labels[idx] = label

        self.labels_ = labels
        return self

    # silhouette score e samples
    def silhouette_samples(self, X):
        if self.labels_ is None:
            raise ValueError("Devi chiamare fit() prima.")

        X = np.asarray(X, dtype=float)
        if self.distance_metric == "cosine":
            X = self._l2_normalize(X)

        labels = self.labels_
        n_samples = len(X)
        unique_labels = np.unique(labels)
        s = np.zeros(n_samples)

        for i in range(n_samples):
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
