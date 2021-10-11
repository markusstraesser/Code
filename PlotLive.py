'''Serial Plotter using PyQTGraph'''

import numpy as np
from numpy.core.fromnumeric import size
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.functions import mkColor
import ArduinoSerial

app = QtGui.QApplication([])

p = pg.plot(title='Dashboard')
p.setAntialiasing(aa=True)
data = np.zeros(1200, dtype='int')
curve = p.plot(pen=mkColor(0, 153, 255))

PORT = "COM4"
SER = ArduinoSerial.createSerial(PORT)

def update():
    global data
    raw = ArduinoSerial.readSerial(SER)
    if raw:
        data[:-1] = data[1:]
        data[-1] = raw[1]
        curve.setData(data)
        app.processEvents()


timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()