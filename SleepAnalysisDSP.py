"""Reads Raw Sensor Data from CSV and performs a Sleep Analysis. Output parameters are: HeartR, RespR, MvtR, SleepPhase"""

from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter, freqz


def getSamplingFreq(timestamp: pd.DataFrame):
    """calculates sampling frequency for a given set of microsecond timestamps"""
    t_start = timestamp.iloc[0]  # ts of first sample
    t_stop = timestamp.iloc[-1]  # ts of last sample
    delta_t = t_stop - t_start  # time difference in microseconds
    # number of samples / time in seconds: sampling frequency in Hz
    fs = len(timestamp) / delta_t * 1000000
    return fs


def getDuration(timestamp: pd.DataFrame):
    t_start = timestamp.iloc[0]  # ts of first sample
    t_stop = timestamp.iloc[-1]  # ts of last sample
    delta_t = int((t_stop - t_start) / 1000000)  # time difference in seconds
    return str(timedelta(seconds=delta_t))


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filter_hr(smoothed_v: pd.DataFrame):
    lowcut = 0.7
    highcut = 3.0
    data["filtered_hr"] = butter_bandpass_filter(
        smoothed_v.to_numpy(dtype=float),
        lowcut,
        highcut,
        fs_sm,
        order=4,
    )


def HeartR(data: pd.DataFrame):
    # TODO write heart rate calculation
    # get sampling frequency
    fs = getSamplingFreq(data["timestamp"])
    hr_vals = data.to_numpy(dtype=float)
    # go through the data and calculate HR for every Minute
    return hr_vals


def filter_rr(smoothed_v: pd.DataFrame):
    lowcut = 0.1
    highcut = 0.66
    data["filtered_rr"] = butter_bandpass_filter(
        smoothed_v.to_numpy(dtype=float),
        lowcut,
        highcut,
        fs_sm,
        order=3,
    )


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


def plot_filt(lower_b, upper_b, param):
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(16, 9)
    tot_duration = getDuration(data["smoothed_ts"])
    sec_duration = getDuration(data["smoothed_ts"].iloc[lower_b:upper_b])
    plt.suptitle(
        f"BCG Sensor Data\nTotal Time: {tot_duration}, Length of Section: {sec_duration}"
    )
    for i in range(3):
        ax[i].set_xlabel("Time (µs)")
        ax[i].set_ylabel("Sensor Value")
        ax[i].ticklabel_format(scilimits=(6, 6), useMathText=True)

    section_width = upper_b - lower_b
    max_y = data["smoothed_v"].max()
    xy_arrow = (
        data["smoothed_ts"].iloc[lower_b],
        data["smoothed_v"].iloc[lower_b],
    )
    xytext = (
        data["smoothed_ts"].iloc[lower_b] + section_width / 2,
        max_y * 0.9,
    )
    ax[0].annotate(
        "Section",
        xy_arrow,
        xytext,
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    ax[0].plot(
        data["smoothed_ts"], data["smoothed_v"], label="Smoothed Data", color="blue"
    )
    ax[0].legend(loc="upper right")
    ax[0].text(
        0,
        data["smoothed_v"].min(),
        f"Sampling Frequency complete Data: {fs:.3f} Hertz\nSampling Frequency complete smoothed Data: {fs_sm:.3f} Hertz",
    )
    ax[1].plot(
        data["smoothed_ts"].iloc[lower_b:upper_b],
        data["smoothed_v"].iloc[lower_b:upper_b],
        label="Smoothed Data Section",
        color="blue",
    )
    ax[1].legend(loc="upper right")
    ax[1].text(
        data["smoothed_ts"].iloc[lower_b:upper_b].min(),
        data["smoothed_v"].iloc[lower_b:upper_b].min(),
        f"Sampling Frequency Section: {fs2:.3f} Hertz",
    )
    # configure subplot to the selected "param": hr, rr, mvt
    ax[2].plot(
        data["smoothed_ts"].iloc[lower_b:upper_b],
        data[f"filtered_{param}"].iloc[lower_b:upper_b],
        label=f"Filtered Data Section {param.upper()}",
        color="orange",
    )
    ax[2].legend(loc="upper right")
    ax[2].text(
        data["smoothed_ts"].iloc[lower_b:upper_b].min(),
        data[f"filtered_{param}"].iloc[lower_b:upper_b].min(),
        f"Sampling Frequency filtered Section: {fs2_sm:.3f} Hertz",
    )
    plt.show()


if __name__ == "__main__":
    # read the csv file into a pandas dataframe
    FILE = "Sensordata\RawData15102021.csv"
    data = pd.read_csv(FILE, names=["timestamp", "value"], delimiter=",")

    print(data)

    # get smoothed dataset
    data["smoothed_ts"] = data["timestamp"].rolling(5, win_type="hanning").mean()
    data["smoothed_v"] = data["value"].rolling(5, win_type="hanning").mean()
    data["smoothed_ts"] = data["smoothed_ts"].bfill()
    data["smoothed_v"] = data["smoothed_v"].bfill()

    # define bounds for section
    sec_low = 1705000
    sec_high = 1708800

    # determine sampling frequencies
    fs = getSamplingFreq(data["timestamp"])
    fs2 = getSamplingFreq(data["timestamp"].iloc[sec_low:sec_high])
    print(f"\nSampling Freq: {fs}\nSampling Freq Section: {fs2}\n")
    fs_sm = float(getSamplingFreq(data["smoothed_ts"]))
    fs2_sm = getSamplingFreq(data["smoothed_ts"].iloc[sec_low:sec_high])

    # filter the smoothed datasets
    filter_rr(data["smoothed_v"])
    filter_hr(data["smoothed_v"])

    print(data)
    # plot
    plot_filt(sec_low, sec_high, "rr")
    plot_filt(sec_low, sec_high, "hr")
