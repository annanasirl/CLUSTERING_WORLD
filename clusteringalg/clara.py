import numpy as np


class CLARA:
    def __init__(self, n_clusters=3, n_samples=5, subset_size=None,
                 distance_metric='euclidean', random_state=None, max_iter=100):

        self.n_clusters = n_clusters                #numero k clusters
        self.n_samples = n_samples                  #n subsets
        self.subset_size = subset_size              #dim subset
        self.distance_metric = distance_metric      #che dist metric usare
        self.random_state = random_state            #x random sampling
        self.max_iter = max_iter                    #max iterazioni pam per subset
        self.medoids = None                         #conterrà i medoids
        self.labels_ = None                         #conterrà i labels

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
        n_points, n_features = X.shape

        # all'inizio, se usiamo distanza cosine, normaliziamo dataset
        # distanza cosine è più adatta per il mio dataset ma le ho implementate tutte e due
        if self.distance_metric == "cosine":
            X = self._l2_normalize(X)

        #se non ho inizializzato subset_size, qua inizializzazione automatica
        #sceglie il minimo tra 40 + 2k e n.punti calcolato prima da shape
        if self.subset_size is None:
            self.subset_size = min(40 + 2 * self.n_clusters, n_points)

        #inizializziamo le variabili che ci serviranno per trovare best medoids e labels
        best_medoids = None
        best_labels = None
        best_cost = np.inf

        #per il numero di samples scelto
        for _ in range(self.n_samples):
            #campiono un sottoinsieme di pt grande "subset size" senza replacement
            subset_idx = np.random.choice(n_points, self.subset_size, replace=False)
            subset = X[subset_idx]

            #inizializzo k medoid casuali nel sottoinsieme
            medoid_idx = np.random.choice(self.subset_size, self.n_clusters, replace=False)
            medoids = subset[medoid_idx].copy()

            #eseguo PAM su sottoinsieme
            for _ in range(self.max_iter):
                #assegna ogni punto del subset al medoid più vicino
                dist_matrix = np.array([self._distance(p, medoids) for p in subset])
                labels_subset = np.argmin(dist_matrix, axis=1)

                #per ogni cluster, seleziono in cluster_points i punti che vi si trovano
                new_medoids = medoids.copy()
                for k in range(self.n_clusters):
                    cluster_points = subset[labels_subset == k]
                    if len(cluster_points) == 0:
                        continue
                    #tra tutti i pt, scelgo quello con la somma di dist dagli altri minima
                    #che diventa nuovo medoids del cluster
                    total_dists = []
                    for p in cluster_points:
                        dists = self._distance(p, cluster_points)
                        total_dists.append(np.sum(dists))
                    total_dists = np.array(total_dists)
                    new_medoids[k] = cluster_points[np.argmin(total_dists)]

                #se i medoids non sono cambiati per una iterazione ho convergenza e finisco
                if np.allclose(new_medoids, medoids):
                    break
                medoids = new_medoids

            #ora assegno al medoid più vicino i pt di tutto il dataset
            dist_matrix_full = np.array([self._distance(p, medoids) for p in X])
            labels_full = np.argmin(dist_matrix_full, axis=1)

            #calcolo tot somma distanze da medoid
            cost = 0
            for i in range(n_points):
                cost += self._distance(X[i], [medoids[labels_full[i]]])[0]
            # se costo minore di best cost allora mi segno medoid e label
            if cost < best_cost:
                best_cost = cost
                best_medoids = medoids.copy()
                best_labels = labels_full.copy()

        #arrivata a questo punto ho in best_medoids e best_labels la miglior configurazione
        self.medoids = best_medoids
        self.labels_ = best_labels


    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.distance_metric == "cosine":
            X = self._l2_normalize(X)
        return np.array([np.argmin(self._distance(x, self.medoids)) for x in X])


    def silhouette_score(self, X):
        X = np.asarray(X, dtype=float)
        if self.distance_metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.maximum(norms, 1e-10)
        n_samples = X.shape[0]
        labels = self.labels_
        score = 0

        for i in range(n_samples):
            own_cluster = X[labels == labels[i]]
            other_clusters = [X[labels == k] for k in np.unique(labels) if k != labels[i]]

            # a(i) = distanza media dal proprio cluster
            if len(own_cluster) > 1:
                a_i = np.mean([self._distance(X[i], [p])[0] for p in own_cluster if not np.array_equal(p, X[i])])
            else:
                a_i = 0

            # b(i) = minima distanza media dagli altri cluster
            b_i = np.min([np.mean([self._distance(X[i], [p])[0] for p in cluster]) for cluster in other_clusters])

            s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
            score += s_i

        return score / n_samples


    def elbow_method(self, X, max_clusters=10):
        X = np.asarray(X, dtype=float)
        if self.distance_metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.maximum(norms, 1e-10)

        inertias = []
        for k in range(1, max_clusters + 1):
            clara = CLARA(n_clusters=k, n_samples=self.n_samples, subset_size=self.subset_size,
                          distance_metric=self.distance_metric,
                          random_state=self.random_state, max_iter=self.max_iter)
            clara.fit(X)
            inertia = np.sum(
                [self._distance(X[i], [clara.medoids[clara.labels_[i]]])[0] ** 2 for i in range(X.shape[0])])
            inertias.append((k, inertia))
        return inertias


    def silhouette_samples(self, X):
        X = np.asarray(X, dtype=float)
        labels = self.labels_
        n_samples = X.shape[0]
        s = np.zeros(n_samples)

        if self.distance_metric == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

        for i in range(n_samples):
            own_cluster = X[labels == labels[i]]
            other_clusters = [X[labels == k] for k in np.unique(labels) if k != labels[i]]

            if len(own_cluster) > 1:
                a_i = np.mean([
                    self._distance(X[i], [p])[0]
                    for p in own_cluster
                    if not np.array_equal(p, X[i])
                ])
            else:
                a_i = 0

            b_i = np.min([
                np.mean([self._distance(X[i], [p])[0] for p in cluster])
                for cluster in other_clusters
            ])

            s[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        return s