"""Simple CSV Plot using PyQTGraph"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

f = "Sensordata\RawData13102021.csv"

# get the Data
data = np.genfromtxt(f, delimiter=",", names=["Timestamp", "Pressure"])


# create pyqtgraph app
app = pg.mkQApp("BCG Data")

win = pg.GraphicsLayoutWidget(show=True, size=(1400, 800), title="Dashboard")
win.setWindowTitle("Dashboard")

# create the plot object
plot1 = win.addPlot(title="Raw Ballistocardiography Data")
# plot the data
plot1.plot(data["Pressure"], pen=pg.mkPen((0, 153, 255), width=2))

# create the GUI-instance of QApplication
if __name__ == "__main__":
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()
