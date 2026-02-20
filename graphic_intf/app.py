import sys
from PyQt6.QtWidgets import QApplication
from startingWindow import StartWindow

app = QApplication(sys.argv)

window = StartWindow()
window.show()

app.exec()