import numpy as np
from itertools import combinations

#clique che trova cluster in sottospazi
class CLIQUE:
    def __init__(self, num_intervals=7, max_subspace_dim=3,
                 min_cluster_fraction=0.05, top_dims=8, verbose=False):

        self.num_intervals = num_intervals                      #in quanti intervalli divido le dimensioni
        self.max_subspace_dim = max_subspace_dim                #max num dim del sottospazio
        self.min_cluster_fraction = min_cluster_fraction        #minima perc di pt sul tot x diventare cluster
        self.top_dims = max(top_dims, max_subspace_dim)         #questo mi serve per trovare le top dim per var.

        self.subspace_clusters_ = {}                            #dizionario sottospazi densi
        self.labels_ = None                                     #array finale etichette

    #calcola indici delle celle
    # min di ogni colonna
    # max di ogni colonna
    # divido range in n. intervalli no divisone 0
    # poi calcola e ritorna indici

    def _compute_cell_indices(self, X):
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        intervals = (X_max - X_min) / self.num_intervals
        intervals[intervals == 0] = 1e-9
        idx = ((X - X_min) / intervals).astype(int)
        return np.clip(idx, 0, self.num_intervals - 1)

    #trova le celle "dense" - prende in input isnieme di indici celle, le dim da considerare, minimo punti x denso
    # prende le colonne del sottospazio dims
    # conta ogni cella quante volte compare
    # ritorna celle compaiono almeno min pts trasformate in tuple

    def _find_dense_cells(self, cell_indices, dims, min_points):
        subspace = cell_indices[:, dims]
        cells, counts = np.unique(subspace, axis=0, return_counts=True)
        return set(map(tuple, cells[counts >= min_points]))

    #aapproccio apriori x generare sottospazi k da candidati densi dim k-1 (prev)
    # candidates è set per evitare duplicati
    # scorre le coppie di sottospazi (ciclo doppio ??)
    # per ogni coppia di sttospazi, li unisce e fa sort indici
    # candidato k se unione ha dimensione k

    def _generate_candidates(self, prev, k):
        candidates = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                union = tuple(sorted(set(prev[i]) | set(prev[j])))
                if len(union) == k:
                    candidates.add(union)
        return candidates

    # unisci in cluster le celle dense

    def _cluster_dense_cells(self, dense_cells):
        #lista finale del clster di celle, set di celle già esplorate che man mano riempio
        clusters = []
        visited = set()

        #scorro tutte le celle in input
        for cell in dense_cells:
            # se cella già visitata, salto
            if cell in visited:
                continue

            #nuovo cluster sulla cella corrente
            cluster = {cell}
            #stack mi serve per visitare le celle vicine (algoritmo tipo flood fill)
            stack = [cell]

            #finchè ho celle da visitare
            while stack:
                #estrai cella dallo stack e segna come visitata
                cur = stack.pop()
                visited.add(cur)

                #per ogni dimensione della cella genero i vicini cambiando indice di +1 e -1
                for d in range(len(cur)):
                    for delta in (-1, 1):
                        neigh = list(cur)
                        neigh[d] += delta
                        #trasformo nuovo vicino in una tupla
                        neigh = tuple(neigh)

                        #se vicino è denso e non nel cluster lo metto nello stack e nel cluster
                        if neigh in dense_cells and neigh not in cluster:
                            cluster.add(neigh)
                            stack.append(neigh)

            #aggiungo il cluster alla lista finale
            clusters.append(cluster)
        return clusters

    ######################### METODO FIT #####################################

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # qua seleziono le top dims dimensioni più variabili
        # calcolo varianza per le dimensioni, prendo le top dim + variabili e taglio X

        if self.top_dims is not None and self.top_dims < n_features:
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-self.top_dims:]
            X = X[:, top_indices]
            n_features = X.shape[1]

        # calcolo min points che sarebbe il n. minimo di punti che voglio avere in un cluster
        # o frazione min_cl_fraction del n. samples o 2
        min_points = max(2, int(self.min_cluster_fraction * n_samples))

        #qui chiamo cell_indices per calcolare gli indici delle celle
        cell_indices = self._compute_cell_indices(X)

        #dizionario sottospazi densi trovati e lista dei cluster trovati
        dense_subspaces = {}
        all_clusters = []

        # primo giro per trovare i cluster a 1 dimensione sulle dim selezionate
        # per ogni dimensione trovo le celle dense e se ce ne sono almeno 2 metto il sottospazio in dense subspaces
        # raggruppo le celle dense in cluster, salvo in all_clusters
        for d in range(n_features):
            dims = (d,)
            dense = self._find_dense_cells(cell_indices, dims, min_points)
            if len(dense) >= 2:
                dense_subspaces[dims] = True
                for c in self._cluster_dense_cells(dense):
                    all_clusters.append((dims, c))


        #giri per trovare cluster da 2 dimensioni fino a max_subspace_dim
        k = 2
        while dense_subspaces and k <= self.max_subspace_dim:
            # in prev metto i sottospazi densi k-1 già trovati, in candidates il risultato di generate_candidates
            # che mi ritorna sottospazi candidati di dimensione k
            # in new_dense voglio mettere i sottospazi densi dim k
            prev = list(dense_subspaces.keys())
            candidates = self._generate_candidates(prev, k)
            new_dense = {}

            for dims in candidates:
                # per ogni sottospaz candidato controllo che anche i sottospazi k-1 che contiene siano densi
                #uso combination per trovarli tutti
                #se anche uno solo non lo è, per apriori posso scartare dims
                if not all(tuple(sorted(s)) in dense_subspaces
                           for s in combinations(dims, k - 1)):
                    continue

                # trovo celle dense in sottospazio candidato
                dense = self._find_dense_cells(cell_indices, dims, min_points)

                #se ne ho trovate almeno 2, sottospazio è denso, lo memorizzo in new_dense
                #poi raggruppo le celle adiacenti in cluster e aggiungo ogni cluster ad all_clusters
                if len(dense) >= 2:
                    new_dense[dims] = True
                    for c in self._cluster_dense_cells(dense):
                        all_clusters.append((dims, c))

            dense_subspaces = new_dense
            k += 1

        # a questo punto in all_clusters ho tutti i cluster che ho trovato in tutte le dimensioni.

        # lista per memorizzare indici dei punti in ogni cluster
        cluster_points = []

        #in all clusters, per ogni cluster di celle, considera il sottospazio corrispondente e da cell_indices
        #che è dove calcoliamo indici celle per X
        #tiro fuori le colonne che corrispondono al sottospazio
        #cerco i punti che appartengono in questo cluster di celle usando maschera booleana
        #trovo indici dei pt che appartengono al cluster
        # mantengo solo i cluster che hanno almeno min:points e li salvo dentro cluster_points

        for dims, cell_cluster in all_clusters:
            sub = cell_indices[:, dims]
            mask = np.array([tuple(r) in cell_cluster for r in sub])
            idx = np.where(mask)[0]

            if len(idx) >= min_points:
                cluster_points.append((dims, idx))

        self.subspace_clusters_ = {dims: idxs for dims, idxs in cluster_points}
        #ordino i cluster per n.pt decrescente in modo che + grandi = assegnati prima
        #cluster_points.sort(key=lambda x: len(x[1]), reverse=True)
        #ordino i cluster per dimensione decrescente e poi n.pt in modo che + grandi = assegnati prima
        cluster_points.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)

        #inizializzo tutte le etichette come outlier
        labels = np.full(n_samples, -1)

        #contatore etichette cluster
        cluster_id = 0

        #per ogni cluster in ordine di grandezza considero i punti non ancora assegnati
        #se sono rimasti più punti di min_points assegno etichetta cluster
        for _, points in cluster_points:
            unassigned = points[labels[points] == -1]
            if len(unassigned) >= min_points:
                labels[unassigned] = cluster_id
                cluster_id += 1

        self.labels_ = labels

        return self

    def _cosine(self, x, y):
        return 1 - np.dot(x, y)

    def _distance(self, x, centroids):
        return np.array([self._cosine(x, c) for c in centroids])

    def silhouette_samples(self, X):
        """
        Calcola i valori di silhouette per ciascun punto sul sottospazio più grande trovato.
        Punti di rumore (-1) ricevono silhouette 0.
        """
        if self.labels_ is None:
            raise ValueError("Devi chiamare fit() prima.")

        X = np.asarray(X, dtype=float)
        labels = self.labels_
        n_samples = len(X)

        # Trova il cluster con il maggior numero di dimensioni (sottospazio più grande)
        if not hasattr(self, "subspace_clusters_") or len(self.subspace_clusters_) == 0:
            raise ValueError("Nessun sottospazio trovato, non posso calcolare silhouette subspace.")

        max_subspace_dims = max(self.subspace_clusters_.keys(), key=lambda k: len(k))
        dims = max_subspace_dims  # tuple delle dimensioni da usare

        # Seleziona solo le colonne del sottospazio più grande
        X_sub = X[:, dims]

        # Normalizzazione per cosine distance
        norms = np.linalg.norm(X_sub, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_normalized = X_sub / norms

        s = np.zeros(n_samples)
        unique_labels = [k for k in np.unique(labels) if k != -1]

        # Pre-calcolo indici cluster
        cluster_indices = {k: np.where(labels == k)[0] for k in unique_labels}

        for i in range(n_samples):
            if labels[i] == -1:
                s[i] = 0
                continue

            own_idx = cluster_indices[labels[i]]
            own_idx = own_idx[own_idx != i]
            if len(own_idx) == 0:
                s[i] = 0
                continue

            # a(i) = distanza media dal proprio cluster
            a_i = np.mean([self._distance(X_normalized[i], X_normalized[j])
                           for j in own_idx])

            # b(i) = minima distanza media dagli altri cluster
            b_i = np.inf
            for k, idxs in cluster_indices.items():
                if k == labels[i]:
                    continue
                dist = np.mean([self._distance(X_normalized[i], X_normalized[j])
                                for j in idxs])
                if dist < b_i:
                    b_i = dist

            if not np.isfinite(a_i) or not np.isfinite(b_i) or max(a_i, b_i) == 0:
                s[i] = 0
            else:
                s[i] = (b_i - a_i) / max(a_i, b_i)

        return s

    def silhouette_score(self, X):
        """
        Media delle silhouette dei punti NON di rumore sul sottospazio più grande.
        """
        s = self.silhouette_samples(X)
        labels = self.labels_
        valid = labels != -1
        if np.sum(valid) == 0:
            return 0
        return np.mean(s[valid])