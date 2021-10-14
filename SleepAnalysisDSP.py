"""Reads Raw Sensor Data from CSV and performs a Sleep Analysis. Output parameters are: HeartR, RespR, MvtR, SleepPhase"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def getSamplingFreq(timestamp: pd.DataFrame):
    """calculates sampling frequency for a given set of microsecond timestamps"""
    t_start = timestamp.iloc[0]  # ts of first sample
    t_stop = timestamp.iloc[-1]  # ts of last sample
    delta_t = t_stop - t_start  # time difference in microseconds
    fs = len(timestamp) / delta_t * 1000000  # sampling frequency in Hz
    return fs

    data.resample(12500).interpolate(method="spline", order=2)


def HeartR(data: pd.DataFrame):
    # TODO write heart rate calculation
    # get sampling frequency
    fs = getSamplingFreq(data["timestamp"])
    hr_vals = data.to_numpy(dtype=float)
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
    # read the csv file into a pandas dataframe
    FILE = "Sensordata\RawData13102021.csv"
    data = pd.read_csv(FILE, names=["timestamp", "value"], delimiter=",")

    print(data)

    # get smoothed dataset
    data["smoothed_ts"] = data["timestamp"].rolling(5, win_type="hanning").mean()
    data["smoothed_v"] = data["value"].rolling(5, win_type="hanning").mean()

    # define bounds for section
    lower = 33050
    upper = 33450

    # determine sampling frequencies
    fs = getSamplingFreq(data["timestamp"])
    fs_2 = getSamplingFreq(data["timestamp"].iloc[lower:upper])
    print(f"\nSampling Freq: {fs}\nSampling Freq Section: {fs_2}\n")

    # plot everything
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(16, 8)
    plt.suptitle("BCG Sensor Data")
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlabel("Time (µs)")
            ax[i, j].set_ylabel("Sensor Value")

    ax[0, 0].plot(data["timestamp"], data["value"], label="Raw Data")
    ax[0, 0].plot(
        data["timestamp"].iloc[lower:upper],
        data["value"].iloc[lower:upper],
        color="orange",
    )
    ax[0, 0].ticklabel_format(scilimits=(6, 6), useMathText=True)
    ax[0, 0].legend(loc="upper right")
    ax[0, 0].text(
        0, data["value"].min(), f"Sampling Frequency complete Data: {fs:.3f} Hertz"
    )
    ax[0, 1].plot(data["smoothed_ts"], data["smoothed_v"], label="Smoothed Data")
    ax[0, 1].plot(
        data["smoothed_ts"].iloc[lower:upper],
        data["smoothed_v"].iloc[lower:upper],
        color="orange",
    )
    ax[0, 1].ticklabel_format(scilimits=(6, 6), useMathText=True)
    ax[0, 1].legend(loc="upper right")
    ax[0, 1].text(
        0,
        data["smoothed_v"].min(),
        f"Sampling Frequency complete smoothed Data: {fs:.3f} Hertz",
    )
    ax[1, 0].plot(
        data["timestamp"].iloc[lower:upper],
        data["value"].iloc[lower:upper],
        label="Raw Data Section",
        color="orange",
    )
    ax[1, 0].legend(loc="upper right")
    ax[1, 0].ticklabel_format(scilimits=(6, 6), useMathText=True)
    ax[1, 0].text(
        data["timestamp"].iloc[lower:upper].min(),
        data["value"].iloc[lower:upper].min(),
        f"Sampling Frequency Section: {fs_2:.3f} Hertz",
    )
    ax[1, 1].plot(
        data["smoothed_ts"].iloc[lower:upper],
        data["smoothed_v"].iloc[lower:upper],
        label="Smoothed Data Section",
        color="orange",
    )
    ax[1, 1].legend(loc="upper right")
    ax[1, 1].ticklabel_format(scilimits=(6, 6), useMathText=True)
    ax[1, 1].text(
        data["smoothed_ts"].iloc[lower:upper].min(),
        data["smoothed_v"].iloc[lower:upper].min(),
        f"Sampling Frequency smoothed Section: {fs_2:.3f} Hertz",
    )
    print(data)
    plt.show()
