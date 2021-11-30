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
import warnings


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


def reject_outliers(data, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    return np.where(s > m, np.nan, data)


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
    """
    Returns heart rate and rmssd for every minute.
    A window of 70 seconds worth of signal is moved in 60 second intervals.
    If heartpy fails to compute valid values, skip one interval.
    """
    lower = 0
    timer = 0
    hr_vals, hrv_vals, timecodes = [], [], []
    # go through the hr data in 60 second steps with a window of length 70 seconds.
    # This means, each window is overlapping the last one by 10 seconds
    for upper in range(int(fs * 70), len(hr), int(fs * 60)):
        try:
            _, m = hp.process(
                hp.scale_data(hr[lower:upper]),
                sample_rate=fs,
                clean_rr=True,
                high_precision=True,
            )
            # write the values to corresponding list
            hr_vals.append(np.round(m["bpm"], 0))
            hrv_vals.append(np.round(m["rmssd"], 0))
            timecodes.append(timer)
        except Exception:
            # if heartpy can't return valid values, skip one 60 second interval
            hr_vals.append(np.nan)
            hrv_vals.append(np.nan)
            timecodes.append(timer)
        # set new lower bound 10 seconds before upper
        lower = upper - int((fs * 10))
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
    """
    Returns respiratory rate and movement for every minute.
    Average distance between peaks in a 1 minute window is calculated for RR,
    min/max values are calculated for movement magnitude.
    A window of 70 seconds worth of signal is moved in 60 second intervals.
    """
    lower = 0
    timer = 0
    rr_vals, mvt_vals, timecodes, diffs = [], [], [], []
    # start at 70 seconds, move in 60 sec intervals
    for upper in range(int(fs * 70), len(rr), int(fs * 60)):
        window = rr[lower:upper]
        # find peaks with following parameters
        peaks, _ = find_peaks(
            window,
            height=np.median(window),
            distance=int(fs * 2),
            width=60,
        )
        # go through the peak values
        for i in range(len(peaks) - 1):
            # and calculate the distance for every consecutive pair
            diffs.append(int(peaks[i + 1] - peaks[i]))
        # calculate average bpm and append to rr_vals
        rr_vals.append(np.round((70 * fs) / np.median(diffs), 1))
        diffs.clear()
        # Calculate magnitude of movement for every minute
        # set min/max to the first value of the window
        min_val = window[0]
        max_val = window[0]
        # go through the window and find min/max values
        for i in window:
            if i < min_val:
                min_val = i
            if i > max_val:
                max_val = i
        # difference between min/max is the movement magnitude
        mvt_vals.append(max_val - min_val)
        # to later make the threshold setting easier, normalize the data
        mvt_vals_norm = [int(i * 100 / max(mvt_vals)) for i in mvt_vals]
        # keep track of the minute
        timecodes.append(timer)
        # set new bounds for window
        lower = upper - int((10 * fs))
        timer += 1
    return rr_vals, mvt_vals_norm, timecodes


def SleepPhases(heart_rates, rmssd, resp_rates, movement):
    """Sleep phases are determined by classifying vital parameters into +/-.

    Median values of a smaller and a larger surrounding window are compared and
    moved through the data.

    Essentially, the parameter curves are lowpass filtered and combined to yield
    the sleep phases.

    ,,,,,,,,,,,,,->|....padding....|---segment---|....padding....|->,,,,,,,,,,

    Inputs:
    - Heart Rates every Minute, Array-like
    - RMSSD (Heart Rate Variability), Array-like
    - Respiratory Rates, Array-like
    - Movement Rates, Array-like

    Outputs:
    - Sleep Phases, Array-like
    - Sleep Stats, Dictionary"""

    lower = 0
    sp_vals = []
    sleep_phase = 0
    length = len(movement)
    step = 1
    window_width = 30
    padding = 75

    # go through the movement data using a window
    for upper in range(window_width, length, step):
        # check if end is reached, calculate the last segment as awake
        if (length - upper) <= step:
            upper = length
            lower = upper - window_width
            sp_vals.extend([2] * (upper - lower))
        else:
            lower = upper - window_width
            # set medians for the window
            hr_avg_segment = int(np.nanmedian(heart_rates[lower:upper]))
            # rmssd_avg_segment = int(np.nanmedian(rmssd[lower:upper]))
            rr_avg_segment = np.nanmedian(resp_rates[lower:upper])
            mvt_avg_segment = int(np.nanmedian(movement[lower:upper]))
            # calculate median of surrounding minutes as threshold for +/- of sleep paramters
            # check if out of bounds and adjust accordingly
            lo = max(lower - padding, 0)
            hi = min(upper + padding, length)
            hr_avg = int(np.nanmedian(heart_rates[lo:hi]))
            # rmssd_avg = int(np.nanmedian(rmssd[lo:hi]))
            rr_avg = np.nanmedian(resp_rates[lo:hi])
            mvt_avg = int(np.nanmean(movement[lo:hi]))
            # determine sleep phase
            if (
                hr_avg_segment >= hr_avg
                and mvt_avg_segment < mvt_avg
                and rr_avg_segment > rr_avg
                # and rmssd_avg_segment > rmssd_avg
            ):
                sleep_phase = 3  # REM
            elif (
                hr_avg_segment < hr_avg
                and mvt_avg_segment < mvt_avg
                and rr_avg_segment < rr_avg
                # and rmssd_avg_segment < rmssd_avg
            ):
                sleep_phase = 1  # DEEP
            elif (
                hr_avg_segment >= hr_avg
                and mvt_avg_segment > mvt_avg
                and rr_avg_segment >= rr_avg
            ):
                sleep_phase = 4  # Awake
            else:
                sleep_phase = 2  # LIGHT
            # set the sleep phase for current segment
            sp_vals.extend([sleep_phase] * (step))

    # stats for sleep duration/quality
    # TODO write sleep stats

    stats = {}
    stats["Duration"] = "{:02d}:{:02d}".format(*divmod(len(sp_vals), 60)) + " h"
    stats["Deep"] = "{:02d}:{:02d}".format(*divmod(sp_vals.count(1), 60)) + " h"
    stats["Light"] = "{:02d}:{:02d}".format(*divmod(sp_vals.count(2), 60)) + " h"
    stats["REM"] = "{:02d}:{:02d}".format(*divmod(sp_vals.count(3), 60)) + " h"
    stats["Interruptions"] = str(0)
    stats["Average HR"] = str(int(np.nanmedian(heart_rates))) + " bpm"
    stats["Average RR"] = str(int(np.nanmedian(resp_rates))) + " bpm"

    return sp_vals, stats


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
    sleep_phases: list,
    sleep_stats: dict,
):
    """Creates Plots for:
    - Complete Raw Dataset
    - Zoomed Sections of Filtered Heartrate and Respiratory Signal
    - Heartrate, HR-Variability, Respiratory Rate and Movement in 1 Minute Sections
    - Sleep Phases"""

    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(17, 8)
    fig.suptitle(f"Sleep Analysis using BCG", fontsize=16)

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

    # set colors for sleep phases
    cc = []
    for val in sleep_phases:
        if val == 1:
            cc.append("#014f86")
        elif val == 2:
            cc.append("#61a5c2")
        elif val == 3:
            cc.append("#a9d6e5")
        else:
            cc.append("lightgrey")

    # plot data
    ax1.plot(raw_timestamps / 60_000_000, raw_data, color="dimgrey")
    ax2.plot(
        # this only works for fixed sample rate of 85.2 Hz, good enough for now
        np.linspace(0, 2575 / fs_sm, 2575),
        # TODO make bounds non static
        rrdata_filt[int(len(rrdata_filt) * 0.4) : int(len(rrdata_filt) * 0.4) + 2575],
        color="#008000",
    )
    ax3.plot(
        # this only works for fixed sample rate of 85.2 Hz, good enough for now
        np.linspace(0, 2575 / fs_sm, 2575),
        # TODO make bounds non static
        hrdata_filt[int(len(hrdata_filt) * 0.4) : int(len(hrdata_filt) * 0.4) + 2575],
        color="#E1701A",
    )
    ax4.bar(rr_section_timestamps, rr_sections, color="#C7E4BD", width=1)
    ax4.plot(rr_section_timestamps, rr_sections, color="#008000")
    ax5.bar(hr_section_timestamps, hr_sections, color="#FFCFAA", width=1)
    ax5.plot(hr_section_timestamps, hr_sections, color="#E1701A")
    ax6.bar(hr_section_timestamps, hrv_sections, color="#F8C8BE", width=1)
    ax6.plot(hr_section_timestamps, hrv_sections, color="#F05C3C")
    ax7.bar(rr_section_timestamps, mvt_sections, color="#E7C9FC", width=1)
    ax7.plot(rr_section_timestamps, mvt_sections, color="#9551C4")
    ax8.bar(np.arange(len(sleep_phases)), sleep_phases, color=cc, width=1)

    # set ticks and labels
    # ax1.set_xticks(np.arange(0, raw_timestamps[-1] / 60_000_000))
    # ax8.set_xticks(np.arange(0, len(sleep_phases)))
    ax8.set_yticks([1.0, 2.0, 3.0, 4.0])
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(6, 6), useMathText=True)
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(3, 3), useMathText=True)
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(3, 3), useMathText=True)
    ax4.set_ylim(bottom=0)
    ax5.set_ylim(bottom=0)
    ax6.set_ylim(bottom=0)
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
    ax7.set_ylabel("Relative Magnitude [%]")
    ax8.set_yticklabels(["Deep", "Light", "REM", "Awake"])

    # Create Textbox for sleep Stats
    props = dict(boxstyle="round", facecolor="lightgrey")
    stats_str = ", ".join("{}: {}".format(k, v) for k, v in sleep_stats.items())
    # place a text box in upper left in axes coords
    ax8.text(
        0.5,
        -0.4,
        stats_str,
        transform=ax8.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=props,
    )

    plt.show()


if __name__ == "__main__":

    # read the csv file into a pandas dataframe
    print("Reading Data from File...")
    FILE = "Sensordata\RawData30112021.csv"
    data = pd.read_csv(FILE, names=["timestamp", "value"], delimiter=",")
    print("Complete!")

    # calculate smoothed dataset and backfill missing data from phaseshift
    data["smoothed_ts"] = data["timestamp"].rolling(5, win_type="hanning").mean()
    data["smoothed_v"] = data["value"].rolling(5, win_type="hanning").mean()
    data["smoothed_ts"] = data["smoothed_ts"].bfill()
    data["smoothed_v"] = data["smoothed_v"].bfill()
    fs_sm = getSamplingFreq(data["smoothed_ts"].to_numpy())

    # filter (and upsample for HR) the smoothed datasets
    print("Filtering Signal...")
    filter_hr(data["smoothed_v"], fs_sm)
    hr_filt = data["filtered_hr"].to_list()
    resampled_hr_filt = resample(hr_filt, len(hr_filt) * 4)
    filter_rr(data["smoothed_v"], fs_sm)
    rr_filt = data["filtered_rr"].to_list()
    print("Filtering done!")

    # Calculate Heartrates, Respiratory Rates, HRV and Movement
    print("Calculating Heart Rates + Heartrate Variability...")
    # ignore warnings thrown by heartpy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heart_rates, hrv, hr_timecodes = HR_HRV(resampled_hr_filt, (fs_sm * 4))
    # remove outliers > +-5 sigma, converts to np.array
    heart_rates = reject_outliers(heart_rates, m=5)
    # interpolate to fill gaps, https://stackoverflow.com/a/6520696
    nans_hr, x_hr = np.isnan(heart_rates), lambda z: z.nonzero()[0]
    heart_rates[nans_hr] = np.round(
        np.interp(x_hr(nans_hr), x_hr(~nans_hr), heart_rates[~nans_hr]), 0
    )
    # needs to be np array
    hrv = np.array(hrv)
    nans_hrv, x_hrv = np.isnan(hrv), lambda z: z.nonzero()[0]
    hrv[nans_hrv] = np.round(
        np.interp(x_hrv(nans_hrv), x_hrv(~nans_hrv), hrv[~nans_hrv]), 0
    )
    print("Heartrates + Heartrate Variability done!")
    print("Calculating Respiration Rates + Movement...")
    resp_rates, movement, rr_timecodes = RR_MVT(rr_filt, fs_sm)
    # remove outliers > +-5 sigma, converts to np.array
    resp_rates = reject_outliers(resp_rates, m=5)
    # interpolate to fill gaps, https://stackoverflow.com/a/6520696
    nans_rr, x_rr = np.isnan(resp_rates), lambda z: z.nonzero()[0]
    resp_rates[nans_rr] = np.round(
        np.interp(x_rr(nans_rr), x_rr(~nans_rr), resp_rates[~nans_rr]), 0
    )
    print("Respiratory Rates + Movement done!")

    # Determine Sleep Phases
    print("Sleep Analysis...")
    sp_segments, sp_stats = SleepPhases(heart_rates, hrv, resp_rates, movement)
    print("Sleep Analysis done!")
    print(sp_stats)

    # Plot everything
    print("Plotting...")
    plot_dash(
        data["smoothed_v"].to_numpy(),
        hr_filt,
        rr_filt,
        heart_rates,
        hrv,
        resp_rates,
        movement,
        data["smoothed_ts"].to_numpy(),
        hr_timecodes,
        rr_timecodes,
        sp_segments,
        sp_stats,
    )
