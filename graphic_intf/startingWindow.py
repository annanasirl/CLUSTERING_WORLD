import sys

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from interfaceClusteringWindow import ClusterWindow
from datasetWindow import DatasetWindow

class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CLUSTERING COUNTRIES")
        self.win_dataset = DatasetWindow()
        self.win_clustering = ClusterWindow()

        buttons = QVBoxLayout()
        buttons.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("CLUSTERING COUNTRIES")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        buttons.addWidget(title)

        description = QLabel("Clustering Countries on Integrated Socio Economic Data")
        description.setAlignment(Qt.AlignmentFlag.AlignLeft)
        buttons.addWidget(description)

        picture = QLabel(self)
        pixmap = QPixmap('../images/images.jpeg')
        picture.setPixmap(pixmap)
        picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        buttons.addWidget(picture)

        button1 = QPushButton("The Dataset")
        button1.setFixedSize(QSize(840, 50))
        buttons.addWidget(button1)

        button2 = QPushButton("Start Clustering")
        button2.setFixedSize(QSize(840, 50))
        buttons.addWidget(button2)

        w = QWidget()
        w.setLayout(buttons)
        self.setCentralWidget(w)

        button1.clicked.connect(self.show_dataset_window)
        button2.clicked.connect(self.show_cluster_window)


    def show_dataset_window(self, checked):
        if self.win_dataset.isVisible():
            self.win_dataset.hide()

        else:
            self.win_dataset.show()

    def show_cluster_window(self, checked):
        if self.win_clustering.isVisible():
            self.win_clustering.hide()

        else:
            self.win_clustering.show()


