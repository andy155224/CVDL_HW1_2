import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

from UI import *

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    Form = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())