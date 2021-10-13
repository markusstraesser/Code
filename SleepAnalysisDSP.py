"""Reads Raw Sensor Data from CSV and performs a Sleep Analysis. Output parameters are: HeartR, RespR, MvtR, SleepPhase"""
import numpy as np
import matplotlib.pyplot as plt


def normalize(data):
    a = 0
    b = max(data)
    norm_data = (data - np.min(data)) * b / (np.max(data) - np.min(data))
    return norm_data


def getSamplingFreq(timestamps):
    """calculates sampling frequenzy for a given set of microsecond timestamps"""
    t_start = timestamps[0]
    t_stop = timestamps[-1]
    delta_t = t_stop - t_start
    fs = len(timestamps) / delta_t * 1000000
    return fs


def HeartR(data):
    # TODO write heart rate calculation
    hr_vals = np.empty(dtype=int)
    # go through the data and calculate HR for every Minute
    return hr_vals


def RespR(data):
    # TODO write respiratory rate calculation
    rr_vals = np.empty(dtype=int)
    # go through the data and calculate RR for every Minute
    return rr_vals


def MvtR(data):
    # TODO write movement rate calculation
    mr_vals = np.empty()
    # go through the data and calculate MR for every Minute

    # Ansatz: Differenz Max-Min Wert des Druckwerts für jede Minute berechnen, dann Schwellenwert für die Nacht bestimmen, wann Bewegung viel und wann wenig ist

    return mr_vals


def SleepPhase():
    # TODO write sleep Phase calculation
    # Ansatz: Parameter Min/Max bestimmen (für alle 3) und dann niedrigen und Hohen Bereich festlegen (einfach zweigeteilt)
    sp_vals = np.empty()
    # go through the data and calculate SP for every Minute
    return sp_vals


if __name__ == "__main__":
    # read the csv file
    FILE = "Sensordata\RawData13102021.csv"
    raw_data = np.genfromtxt(
        FILE, dtype=(float, float), names=["timestamps", "value"], delimiter=","
    )
    print("Raw Data:", raw_data)
    norm_data = raw_data
    norm_data["value"] = normalize(raw_data["value"])
    print("Normalized Data:", norm_data)

    # sampling frequenzy
    fs = getSamplingFreq(norm_data["timestamps"])
    print(fs)

    plt.figure(figsize=(16, 5))
    plt.plot(norm_data["timestamps"], norm_data["value"])
    plt.text(0, 0, "Sampling Frequenzy: %.3f Hertz" % fs)
    plt.show()
