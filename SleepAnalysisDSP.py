"""Reads Raw Sensor Data from CSV and performs a Sleep Analysis. Output parameters are: HeartR, RespR, MvtR, SleepPhase"""

from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter
import heartpy as hp
from scipy.signal import resample, find_peaks
import matplotlib.gridspec as gridspec


def getSamplingFreq(timestamp):
    """calculates sampling frequency for a given set of microsecond timestamps"""
    t_start = timestamp[0]  # ts of first sample
    t_stop = timestamp[-1]  # ts of last sample
    delta_t = t_stop - t_start  # time difference in microseconds
    # number of samples / time in seconds: sampling frequency in Hz
    fs = len(timestamp) / delta_t * 1_000_000
    return fs


def getDuration(timestamp):
    t_start = timestamp[0]  # ts of first sample
    t_stop = timestamp[-1]  # ts of last sample
    delta_t = int((t_stop - t_start) / 1_000_000)  # time difference in seconds
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


def filter_hr(smoothed_v: pd.DataFrame, fs):
    lowcut = 0.7
    highcut = 2.7
    data["filtered_hr"] = butter_bandpass_filter(
        smoothed_v.to_numpy(dtype=float),
        lowcut,
        highcut,
        fs,
        order=5,
    )


def HR_HRV(hr: list, fs):
    lower = 0
    upper = int(fs * 60)
    print(f"Sampling Frequency Resampled: {fs:.3f}")
    timer = 0
    hr_vals, hrv_vals, timecodes = [], [], []
    # print(len(ts))
    while upper <= len(hr):
        try:
            wd, m = hp.process(
                hr[lower:upper],
                fs,
                clean_rr=True,
            )
            hr_vals.append(m["bpm"])
            hrv_vals.append(m["rmssd"])
            timecodes.append(timer)
        except:
            hr_vals.append(0)
            hrv_vals.append(0)
            timecodes.append(timer)
        lower = upper
        upper += int(fs * 60)
        timer += 1
    return hr_vals, hrv_vals, timecodes


def filter_rr(smoothed_v: pd.DataFrame, fs):
    lowcut = 0.1
    highcut = 0.66
    data["filtered_rr"] = butter_bandpass_filter(
        smoothed_v.to_numpy(dtype=float),
        lowcut,
        highcut,
        fs,
        order=3,
    )


def RespR(rr: list, fs):
    lower = 0
    upper = int(fs * 30)
    print(f"Sampling Frequency Resampled: {fs:.3f}")
    timer = 0
    rr_vals, timecodes = [], []
    # print(len(ts))
    while upper <= len(hr):
        try:
            wd, m = hp.process(
                rr[lower:upper],
                fs,
                clean_rr=True,
            )
            hp.process
            hr_vals.append(m["bpm"])
            timecodes.append(timer)
        except:
            timecodes.append(timer)
            hr_vals.append(0)
        lower = upper
        upper += int(fs * 30)
        timer += 0.5
    return hr_vals, timecodes


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


# all the plotting stuff
def plot_dash(
    rawdata,
    hrdata_filt,
    rrdata_filt,
    hr_sections,
    hrv_sections,
    # rr_sections,
    # mvt_sections,
    section_timestamps
    # sleep_phases,
):
    """Creates Plots for:
    - Complete Raw Dataset
    - Zoomed Sections of Filtered Heartrate and Respiratory Signal
    - Heartrate, HR-Variability, Respiratory Rate and Movement in 60 Second Sections
    - Sleep Phases"""

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(18, 8)
    figmgr = plt.get_current_fig_manager()
    figmgr.window.showMaximized()
    fig.suptitle(
        f"Sleep Analysis using BCG (Total Recorded Time: {tot_duration})", fontsize=16
    )

    gs = gridspec.GridSpec(nrows=6, ncols=4)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :2])
    ax3 = fig.add_subplot(gs[1, 2:])
    ax4 = fig.add_subplot(gs[2:4, 0])
    ax5 = fig.add_subplot(gs[2:4, 1])
    ax6 = fig.add_subplot(gs[2:4, 2])
    ax7 = fig.add_subplot(gs[2:4, 3])
    ax8 = fig.add_subplot(gs[-2:, :])

    ax1.plot(rawdata)
    ax1.set_title("Raw Sensor Data")
    ax2.plot(rrdata_filt[250000:252575])
    ax2.set_title("Filtered Respiratory Signal")
    ax3.plot(hrdata_filt[250000:252575])
    ax3.set_title("Filtered Heartrate Signal")
    ax4.set_title("Respiratory Rate in 1 Minute Intervals")
    ax5.bar(section_timestamps, hr_sections, width=1)
    ax5.set_title("Heart Rate in 1 Minute Intervals")
    ax6.bar(section_timestamps, hrv_sections, width=1)
    ax6.set_title("Heart Rate Variability in 1 Minute Intervals")
    ax7.set_title("Movement in 1 Minute Intervals")
    ax8.set_title("Sleep Phases")

    plt.show()


if __name__ == "__main__":
    # read the csv file into a pandas dataframe
    FILE = "Sensordata\RawData15102021.csv"
    data = pd.read_csv(FILE, names=["timestamp", "value"], delimiter=",")

    # calculate smoothed dataset and backfill
    data["smoothed_ts"] = data["timestamp"].rolling(5, win_type="hanning").mean()
    data["smoothed_v"] = data["value"].rolling(5, win_type="hanning").mean()
    data["smoothed_ts"] = data["smoothed_ts"].bfill()
    data["smoothed_v"] = data["smoothed_v"].bfill()
    fs_sm = getSamplingFreq(data["smoothed_ts"].to_numpy())
    print(f"Sampling Frequency: {fs_sm:.3f}")

    tot_duration = getDuration(data["smoothed_ts"].to_list())

    # filter and upsample the smoothed datasets
    filter_hr(data["smoothed_v"], fs_sm)
    hr_filt = data["filtered_hr"].to_list()
    res_hr_filt = resample(hr_filt, len(hr_filt) * 2)
    filter_rr(data["smoothed_v"], fs_sm)
    rr_filt = data["filtered_rr"].to_list()

    # Calculate Heartrates, Respiratory Rates, HRV and Movement for 30 Second Intervals
    heartrates, rmssd, hr_timecodes = HR_HRV(res_hr_filt, (fs_sm * 2))

    # Plot everything
    plot_dash(data["value"], hr_filt, rr_filt, heartrates, rmssd, hr_timecodes)
