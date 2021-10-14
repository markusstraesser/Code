"""Reads Raw Sensor Data from CSV and performs a Sleep Analysis. Output parameters are: HeartR, RespR, MvtR, SleepPhase"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter, freqz


def getSamplingFreq(timestamp: pd.DataFrame):
    """calculates sampling frequency for a given set of microsecond timestamps"""
    t_start = timestamp.iloc[0]  # ts of first sample
    t_stop = timestamp.iloc[-1]  # ts of last sample
    delta_t = t_stop - t_start  # time difference in microseconds
    fs = len(timestamp) / delta_t * 1000000  # sampling frequency in Hz
    return fs


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


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
    data["smoothed_ts"] = data["smoothed_ts"].bfill()
    data["smoothed_v"] = data["smoothed_v"].bfill()

    # define bounds for section
    lower = 18000
    upper = 27000

    # determine sampling frequencies
    fs = getSamplingFreq(data["timestamp"])
    fs2 = getSamplingFreq(data["timestamp"].iloc[lower:upper])
    print(f"\nSampling Freq: {fs}\nSampling Freq Section: {fs2}\n")
    fs_sm = float(getSamplingFreq(data["smoothed_ts"]))
    fs2_sm = getSamplingFreq(data["smoothed_ts"].iloc[lower:upper])

    # filter the two smoothed datasets

    # TODO create filter scripts for hr, rr, mvt
    lowcut = 0.0933
    highcut = 0.266
    data["filtered_v"] = butter_bandpass_filter(
        data["smoothed_v"].to_numpy(dtype=float),
        lowcut,
        highcut,
        fs_sm,
        order=3,
    )
    print(data)
    # plot everything
    fig, ax = plt.subplots(4, 1)
    fig.set_size_inches(16, 8)
    plt.suptitle("BCG Sensor Data")
    for i in range(4):
        ax[i].set_xlabel("Time (µs)")
        ax[i].set_ylabel("Sensor Value")

    ax[0].plot(
        data["smoothed_ts"], data["smoothed_v"], label="Smoothed Data", color="blue"
    )

    ax[0].ticklabel_format(scilimits=(6, 6), useMathText=True)
    ax[0].legend(loc="upper right")
    ax[0].text(
        0,
        data["smoothed_v"].min(),
        f"Sampling Frequency complete Data: {fs:.3f} Hertz\nSampling Frequency complete smoothed Data: {fs_sm:.3f} Hertz",
    )
    ax[1].plot(
        data["smoothed_ts"], data["filtered_v"], label="Filtered Data", color="orange"
    )
    ax[2].plot(
        data["smoothed_ts"].iloc[lower:upper],
        data["smoothed_v"].iloc[lower:upper],
        label="Smoothed Data Section",
        color="blue",
    )
    ax[2].legend(loc="upper right")
    ax[2].ticklabel_format(scilimits=(6, 6), useMathText=True)
    ax[2].text(
        data["smoothed_ts"].iloc[lower:upper].min(),
        data["smoothed_v"].iloc[lower:upper].min(),
        f"Sampling Frequency Section: {fs2:.3f} Hertz\nSampling Frequency smoothed Section: {fs2_sm:.3f} Hertz",
    )
    ax[3].plot(
        data["smoothed_ts"].iloc[lower:upper],
        data["filtered_v"].iloc[lower:upper],
        label="Smoothed Data Section",
        color="orange",
    )
    print(data)
    plt.show()
