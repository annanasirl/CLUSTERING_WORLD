from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QTextEdit, QSplitter
from matplotlib import cm
import pandas as pd
import numpy as np

class ClusterStatsWindow(QDialog):
    def __init__(self, df, labels, dispersion_func='std'):
        """
        Tre bande verticali ridimensionabili:
        1) Tabella completa: feature x cluster, con media ± dispersione.
        2) Tabella abbreviazioni: solo media/alta/bassa e uniforme.
        3) Box con descrizioni testuali complete.
        """
        super().__init__()
        self.setWindowTitle("Statistiche cluster")
        self.resize(1200, 900)

        # --- Prepara dati ---
        feature_cols = df.columns[2:-1]
        df_feat = df[feature_cols].copy()
        df_feat["cluster"] = labels
        df_feat = df_feat[df_feat["cluster"] != -1]

        if df_feat.empty:
            return

        cluster_means = df_feat.groupby("cluster").mean()
        if dispersion_func == 'std':
            cluster_disp = df_feat.groupby("cluster").std()
        elif dispersion_func == 'iqr':
            cluster_disp = df_feat.groupby("cluster").quantile(0.75) - df_feat.groupby("cluster").quantile(0.25)
        else:
            raise ValueError("dispersion_func deve essere 'std' o 'iqr'")

        clusters = list(cluster_means.index)
        features = list(cluster_means.columns)

        # --- Splitter verticale ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        # --- 1) Tabella completa ---
        table = QTableWidget()
        splitter.addWidget(table)
        table.setRowCount(len(features))
        table.setColumnCount(len(clusters))
        table.setHorizontalHeaderLabels([str(c) for c in clusters])
        table.setVerticalHeaderLabels(features)

        all_vals = cluster_means.values.flatten()
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        cmap = cm.get_cmap("coolwarm_r")

        font = QFont()
        font.setBold(False)
        font.setPointSize(10)

        for row_idx, feature in enumerate(features):
            for col_idx, cluster_id in enumerate(clusters):
                mean_val = cluster_means.loc[cluster_id, feature]
                disp_val = cluster_disp.loc[cluster_id, feature]
                text = f"{mean_val:.3f} ± {disp_val:.3f}"

                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFont(font)
                item.setForeground(QColor("black"))

                norm_val = (mean_val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                rgba = cmap(norm_val)
                item.setBackground(QColor.fromRgbF(rgba[0], rgba[1], rgba[2], rgba[3]))

                table.setItem(row_idx, col_idx, item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        # --- 2) Tabella abbreviazioni ---
        abbrev_table = QTableWidget()
        splitter.addWidget(abbrev_table)
        abbrev_table.setColumnCount(len(clusters))
        abbrev_table.setHorizontalHeaderLabels([str(c) for c in clusters])

        # Generiamo liste abbreviate
        abbrev_data = []
        max_rows = 0
        for cluster_id in clusters:
            col_list = []
            for feature in features:
                mean_val = cluster_means.loc[cluster_id, feature]
                disp_val = cluster_disp.loc[cluster_id, feature]

                # media livello
                if mean_val > 0.5:
                    level = "ALTA"
                elif mean_val < -0.5:
                    level = "BASSA"
                else:
                    level = "MEDIA"

                # dispersione
                uniform = disp_val < 0.5

                # solo condizioni richieste
                if level in ["ALTA", "BASSA"] and uniform:
                    col_list.append(f"{feature}: {level}")

            abbrev_data.append(col_list)
            max_rows = max(max_rows, len(col_list))

        # Impostiamo righe in base alla colonna più lunga
        abbrev_table.setRowCount(max_rows)
        for col_idx, col_list in enumerate(abbrev_data):
            for row_idx, text in enumerate(col_list):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
                abbrev_table.setItem(row_idx, col_idx, item)

        abbrev_table.resizeColumnsToContents()
        abbrev_table.resizeRowsToContents()

        # --- 3) Box descrizioni complete ---
        text_box = QTextEdit()
        text_box.setReadOnly(True)
        splitter.addWidget(text_box)

        descriptions = []
        for cluster_id in clusters:
            desc_lines = [f"Cluster {cluster_id}:"]
            for feature in features:
                mean_val = cluster_means.loc[cluster_id, feature]
                disp_val = cluster_disp.loc[cluster_id, feature]

                if mean_val > 0.5:
                    level = "alto"
                elif mean_val < -0.5:
                    level = "basso"
                else:
                    level = "medio"

                if disp_val < 0.5:
                    uniformity = "uniforme"
                elif disp_val > 1.2:
                    uniformity = "disomogeneo"
                else:
                    uniformity = "variegato"

                desc_lines.append(f"{feature}: {level}, {uniformity}")
            descriptions.append("\n".join(desc_lines))

        text_box.setText("\n\n".join(descriptions))

        # --- Impostiamo proporzioni iniziali ---
        splitter.setSizes([400, 200, 300])
