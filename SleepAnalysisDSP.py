"""Reads Raw Sensor Data from CSV and performs a Sleep Analysis.
Output parameters are: HeartR, HRV, RespR, MvtR, SleepPhase"""

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
    lowcut = 0.66
    highcut = 2.33
    data["filtered_hr"] = butter_bandpass_filter(
        smoothed_v.to_numpy(dtype=float),
        lowcut,
        highcut,
        fs,
        order=4,
    )


def HR_HRV(hr: list, fs):
    lower = 0
    # upper = int(fs * 70)
    timer = 0
    hr_vals, hrv_vals, timecodes = [], [], []
    for upper in range(int(fs * 70), len(hr), int(fs * 60)):
        try:
            _, m = hp.process(
                hp.scale_data(hr[lower:upper]),
                sample_rate=fs,
                clean_rr=True,
                high_precision=True,
            )
            hr_vals.append(m["bpm"])
            hrv_vals.append(m["rmssd"])
            timecodes.append(timer)
        except:
            pass
        lower = upper - 10
        # upper += int(fs * 60)
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


def RR_MVT(rr: list, fs):
    lower = 0
    upper = int(fs * 70)
    timer = 0
    rr_vals, mvt_vals, timecodes, diffs = [], [], [], []
    while upper <= len(rr):
        window = rr[lower:upper]
        peaks, _ = find_peaks(window, height=0, distance=fs * 2, width=60)
        for i in range(len(peaks) - 1):
            diffs.append(peaks[i + 1] - peaks[i])
        rr_vals.append((70 * fs) / np.mean(diffs))
        diffs.clear()
        # Ansatz: Differenz Max-Min Wert des Druckwerts für jede
        # Minute berechnen
        min_val = window[0]
        max_val = window[0]
        for i in window:
            if i < min_val:
                min_val = i
            if i > max_val:
                max_val = i
        mvt_vals.append(max_val - min_val)
        mvt_vals_norm = [float(i) / max(mvt_vals) for i in mvt_vals]
        timecodes.append(timer)
        lower = upper - 10
        upper += int(fs * 60)
        timer += 1
    return rr_vals, mvt_vals_norm, timecodes


def SleepPhases(heart_rates, rmssd, resp_rates, movement):
    """ "Movement (e.g. turning in Bed) is used to segment the night"""
    # TODO larger segmentation into sleep cycles of 90-120 minutes for "_avg" values
    # This should enable more accurate classification of sleep phases by
    # creating medians for every sleep cycle, not for the whole night

    lower = 0
    upper = 5
    threshold = 0.2
    sp_vals = []
    sleep_phase = 0
    # Determine medians as threshold for +/- of sleep paramters
    hr_avg = np.nanmedian(heart_rates)
    rmssd_avg = np.nanmedian(rmssd)
    rr_avg = np.nanmedian(resp_rates)
    length = len(movement)

    # go through the movement dataset
    while upper < length:
        # look for next position where movement < threshold
        if movement[upper] > threshold:
            while upper < length and movement[upper] > threshold:
                upper += 1
        # find next threshold crossing
        while upper < length and movement[upper] < threshold:
            upper += 1
        # calculate median for segment
        hr_avg_segment = np.nanmedian(heart_rates[lower:upper])
        rmssd_avg_segment = np.nanmedian(rmssd[lower:upper])
        rr_avg_segment = np.nanmedian(resp_rates[lower:upper])

        # determine sleep phase
        # TODO determine wake phases
        if (
            hr_avg_segment > hr_avg
            and rmssd_avg_segment > rmssd_avg
            and rr_avg_segment > rr_avg
        ):
            sleep_phase = 3  # REM
        elif (
            hr_avg_segment < hr_avg
            and rmssd_avg_segment < rmssd_avg
            and rr_avg_segment < rr_avg
        ):
            sleep_phase = 1  # DEEP
        else:
            sleep_phase = 2  # LIGHT
        # set the sleep phase for current segment
        sp_vals.extend([sleep_phase] * (upper - lower))
        # set new bounds
        lower = upper
        # increment upper by at least 1
        upper += 1

    # stats for sleep duration/quality
    # TODO write summary of sleep stats
    measures = dict

    return sp_vals, measures


# all the plotting stuff
def plot_dash(
    raw_data,
    hrdata_filt,
    rrdata_filt,
    hr_sections,
    hrv_sections,
    rr_sections,
    mvt_sections,
    raw_timestamps,
    hr_section_timestamps,
    rr_section_timestamps,
    sleep_phases,
    # sleep_measures,
):
    """Creates Plots for:
    - Complete Raw Dataset
    - Zoomed Sections of Filtered Heartrate and Respiratory Signal
    - Heartrate, HR-Variability, Respiratory Rate and Movement in 1 Minute Sections
    - Sleep Phases"""

    plt.style.use("bmh")
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(18, 8)
    figmgr = plt.get_current_fig_manager()
    figmgr.window.showMaximized()
    fig.suptitle(
        f"Sleep Analysis using BCG (Total Recorded Time: {tot_duration})", fontsize=16
    )

    # create layout
    gs = gridspec.GridSpec(nrows=6, ncols=4)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :2])
    ax3 = fig.add_subplot(gs[1, 2:])
    ax4 = fig.add_subplot(gs[2:4, 0])
    ax5 = fig.add_subplot(gs[2:4, 1])
    ax6 = fig.add_subplot(gs[2:4, 2])
    ax7 = fig.add_subplot(gs[2:4, 3])
    ax8 = fig.add_subplot(gs[-2:, :])

    # set titles
    ax1.set_title("Raw Sensor Data")
    ax2.set_title("Section of Filtered Respiratory Signal")
    ax3.set_title("Section of Filtered Heartrate Signal")
    ax4.set_title("Average Respiratory Rate per Minute")
    ax5.set_title("Average Heart Rate per Minute")
    ax6.set_title("Average Heart Rate Variability per Minute")
    ax7.set_title("Average Movement per Minute")
    ax8.set_title("Sleep Phases")

    # plot data
    ax1.plot(raw_timestamps / 60_000_000, raw_data, color="dimgrey")
    ax2.plot(
        np.linspace(0, 2575 / fs_sm, 2575),
        rrdata_filt[1551000:1553575],
        color="goldenrod",
    )
    ax3.plot(
        np.linspace(0, 2575 / fs_sm, 2575), hrdata_filt[250000:252575], color="seagreen"
    )
    ax4.scatter(rr_section_timestamps, rr_sections, color="goldenrod", s=10)
    ax5.scatter(hr_section_timestamps, hr_sections, color="seagreen", s=10)
    ax6.scatter(hr_section_timestamps, hrv_sections, color="teal", s=10)
    ax7.bar(rr_section_timestamps, mvt_sections, color="darkorange", width=2)
    ax8.plot(range(len(sleep_phases)), sleep_phases, color="cornflowerblue")

    # set ticks and labels
    ax1.set_xticks(np.arange(0, max(raw_timestamps) / 60_000_000, 30))
    ax8.set_xticks(np.arange(0, len(sleep_phases), 30))
    ax8.set_yticks([1.0, 2.0, 3.0, 4.0])
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(6, 6), useMathText=True)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(3, 3), useMathText=True)
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(3, 3), useMathText=True)
    ax1.set_xlabel("Time [Minutes]")
    ax2.set_xlabel("Time [Seconds]")
    ax3.set_xlabel("Time [Seconds]")
    ax4.set_xlabel("Time [Minutes]")
    ax5.set_xlabel("Time [Minutes]")
    ax6.set_xlabel("Time [Minutes]")
    ax7.set_xlabel("Time [Minutes]")
    ax8.set_xlabel("Time [Minutes]")
    ax1.set_ylabel("Magnitude")
    ax2.set_ylabel("Magnitude")
    ax3.set_ylabel("Magnitude")
    ax4.set_ylabel("Respiratory Rate [bpm]")
    ax5.set_ylabel("Heart Rate [bpm]")
    ax6.set_ylabel("RMSSD [ms]")
    ax7.set_ylabel("Relative Magnitude")
    ax8.set_yticklabels(["Deep", "Light", "REM", "Awake"])

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

    # calculate duration of recorded data
    tot_duration = getDuration(data["smoothed_ts"].to_list())

    # filter (and upsample for HR) the smoothed datasets
    filter_hr(data["smoothed_v"], fs_sm)
    hr_filt = data["filtered_hr"].to_list()
    resampled_hr_filt = resample(hr_filt, len(hr_filt) * 4)
    filter_rr(data["smoothed_v"], fs_sm)
    rr_filt = data["filtered_rr"].to_list()
    print("Filtering done!")

    # Calculate Heartrates, Respiratory Rates, HRV and Movement
    heart_rates, hrv, hr_timecodes = HR_HRV(resampled_hr_filt, (fs_sm * 4))
    print("HR, Hrv done!")
    resp_rates, movement, rr_timecodes = RR_MVT(rr_filt, fs_sm)
    print("Resp + Mvt done!")

    # Determine Sleep Phases
    sp_segments, _ = SleepPhases(heart_rates, hrv, resp_rates, movement)
    print("Sleep Phases done!")

    # Plot everything
    plot_dash(
        data["smoothed_v"],
        hr_filt,
        rr_filt,
        heart_rates,
        hrv,
        resp_rates,
        movement,
        data["smoothed_ts"],
        hr_timecodes,
        rr_timecodes,
        sp_segments,
        # sp_measures,
    )
