"""Serial Plotter using PyQTGraph"""

import numpy as np
from numpy.core.fromnumeric import size
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.functions import mkColor
import ArduinoSerial

app = QtGui.QApplication([])


p = pg.plot(title="Dashboard")
p.setAntialiasing(aa=True)
p.setLabel("left", "Value")
p.setLabel("bottom", "Samples")

# set the number of samples in the window
data = np.zeros(800)

# add Plot
curve = p.plot(pen=mkColor(51, 102, 255))

p.showMaximized()

PORT = "COM4"
SER = ArduinoSerial.createSerial(PORT)

# get the offset of the pressure values
tare = ArduinoSerial.tare(SER, 20)


def update():
    global data
    raw = ArduinoSerial.readSerial(SER)
    if raw:
        data[:-1] = data[1:]

        # subtract the offset for every new value to zero the Plot
        data[-1] = raw[1] - tare
        curve.setData(data)
        app.processEvents()


timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == "__main__":
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()
