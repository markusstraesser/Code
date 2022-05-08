"""Reads Raw Sensor Data from CSV and performs a Sleep Analysis.
Output parameters are: List of HeartR, HRV, RespR, MvtR, SleepPhase,
SleepPhase durations, avg. HR, RR"""

from datetime import timedelta
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import butter, lfilter
import heartpy as hp
from scipy.signal import resample, find_peaks
import matplotlib.gridspec as gridspec
import warnings


def get_sampling_freq(timestamp):
    """calculates sampling frequency for a given set of microsecond timestamps"""
    t_start = timestamp[0]  # ts of first sample
    t_stop = timestamp[-1]  # ts of last sample
    delta_t = t_stop - t_start  # time difference in microseconds
    # number of samples / time in seconds: sampling frequency in Hz
    fs = len(timestamp) / delta_t * 1_000_000
    return fs


def get_duration(timestamp):
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


def group_list(raw_list: list):
    # TODO if last element has width of 1, it is ignored. Some mistake in the logic...

    """Returns
    - list of unique values in order of appearance
    - list with the number of consecutive occurences
    - positions of chunks (x coordinate of middle)

    Example:
    [2, 2, 2, 1, 3, 3, 4, 4, 4, 4, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 2, 2]

    ->
    [2, 1, 3, 4, 2, 3, 1, 2] [3, 1, 2, 4, 3, 2, 5, 2] [1.5, 3.5, 5.0, 8.0, 11.5,
    14.0, 17.5, 21.0]"""

    grouped_vals = []
    val_count = []
    count = 1
    x_pos = []
    for i in range(0, len(raw_list) - 1):
        current = raw_list[i]
        next_element = raw_list[i + 1]
        if i == len(raw_list) - 2:
            count += 1
            grouped_vals.append(current)
            val_count.append(count)
            x_pos.append(i + 1 - (count / 2) + 1)
        elif current != next_element:
            grouped_vals.append(current)
            val_count.append(count)
            x_pos.append(i - (count / 2) + 1)
            count = 1
        else:
            count += 1
    return grouped_vals, val_count, x_pos


def filter_bp(l, h, smoothed_v: pd.DataFrame, fs):
    lowcut = l
    highcut = h
    bp_filter_output = butter_bandpass_filter(
        smoothed_v.to_numpy(dtype=float),
        lowcut,
        highcut,
        fs,
        order=4,
    )
    return bp_filter_output


def filter_hr(smoothed_v: pd.DataFrame, fs):
    lowcut = 0.66
    highcut = 2.0
    hr_filter_output = butter_bandpass_filter(
        smoothed_v.to_numpy(dtype=float),
        lowcut,
        highcut,
        fs,
        order=4,
    )
    return hr_filter_output


def hr_hrv(hr, fs):
    """
    Returns heart rate and rmssd for every minute.
    A window of 70 seconds worth of signal is moved in 60 second intervals.
    If heartpy fails to compute valid values, skip one interval.
    """
    lower = 0
    timer = 0
    failed = 0
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
            hr_vals.append(np.round(m["bpm"], 1))
            hrv_vals.append(np.round(m["rmssd"], 0))
            timecodes.append(timer)
        except Exception:
            # if heartpy can't return valid values, skip one 60 second interval
            failed += 1
            hr_vals.append(np.nan)
            hrv_vals.append(np.nan)
            timecodes.append(timer)
        # set new lower bound 10 seconds before upper
        lower = upper - int((fs * 10))
        timer += 1
    # # quick fix for 02122021
    # hr_vals.append(np.nan)
    # timecodes.append(timer + 1)
    failure_rate = failed / timer
    return hr_vals, hrv_vals, timecodes, failure_rate


def filter_rr(smoothed_v: pd.DataFrame, fs):
    lowcut = 0.1
    highcut = 0.5
    rr_filter_output = butter_bandpass_filter(
        smoothed_v.to_numpy(dtype=float),
        lowcut,
        highcut,
        fs,
        order=3,
    )
    return rr_filter_output


def rr_mvt(rr, fs):
    """
    Returns respiratory rate and movement for every minute.
    Average distance between peaks in a 1 minute window is calculated for RR,
    min/max values are calculated for movement magnitude.
    A window of 70 seconds worth of signal is moved in 60 second intervals.
    """
    # TODO Write better RR logic to account for raw signal problems
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
        # normalize the data in range 10;
        # this works as a good classification because of how the averaging in
        # the sleep phase calculation ist done.
        mvt_vals_norm = [int(i * 10 / max(mvt_vals)) for i in mvt_vals]
        # keep track of the minute
        timecodes.append(timer)
        # set new bounds for window
        lower = upper - int((10 * fs))
        timer += 1
    failed = np.count_nonzero(np.isnan(rr_vals))
    failure_rate = failed / timer
    return rr_vals, mvt_vals_norm, timecodes, failure_rate


def sleep_phases(heart_rates, rmssd, resp_rates, movement):
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
    window_width = 10
    padding = 50

    hr_averaged_plot = []
    rmssd_averaged_plot = []
    rr_averaged_plot = []
    mvt_averaged_plot = []
    hr_reference_plot = []
    rmssd_reference_plot = []
    rr_reference_plot = []
    mvt_reference_plot = []

    # go through the movement data using a window of the sensordata
    for upper in range(window_width, length, step):
        # check if end is reached, calculate the last segment as awake
        if (length - upper) <= step:
            upper = length
            lower = upper - window_width
            sp_vals.extend([2] * (upper - lower))
        else:
            lower = upper - window_width
            # set median values for the window: this is the (smoothed) sensordata
            hr_avg_segment = np.nanmedian(heart_rates[lower:upper])
            rmssd_avg_segment = int(np.nanmedian(rmssd[lower:upper]))
            rr_avg_segment = np.nanmedian(resp_rates[lower:upper])
            mvt_avg_segment = np.nanmean(movement[lower:upper])
            # calculate median of surrounding minutes as threshold for +/- of sleep paramters;
            # this is what the sensordata is beeing compared to
            # check if out of bounds and adjust accordingly
            lo = max(lower - padding, 0)
            hi = min(upper + padding, length)
            hr_avg = np.nanmedian(heart_rates[lo:hi])
            rmssd_avg = int(np.nanmedian(rmssd[lo:hi]))
            rr_avg = np.nanmedian(resp_rates[lo:hi])
            # movement is averaged over the whole night
            mvt_avg = np.nanmean(movement) * 2.5
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
                hr_avg_segment > hr_avg
                and mvt_avg_segment > mvt_avg
                and rr_avg_segment >= rr_avg
            ):
                sleep_phase = 4  # Awake
            else:
                sleep_phase = 2  # LIGHT
            # set the sleep phase for current segment
            sp_vals.extend([sleep_phase] * (step))

            # write the median sensordata of the window into a list
            hr_averaged_plot.append(hr_avg_segment)
            rmssd_averaged_plot.append(rmssd_avg_segment)
            rr_averaged_plot.append(rr_avg_segment)
            mvt_averaged_plot.append(mvt_avg_segment)
            hr_reference_plot.append(hr_avg)
            rmssd_reference_plot.append(rmssd_avg)
            rr_reference_plot.append(rr_avg)
            mvt_reference_plot.append(mvt_avg)

    # quick fix for lenght issue (1 missing)
    sp_vals.extend([sleep_phase])

    # stats for sleep duration/quality
    sp_sequence, _, _ = group_list(
        sp_vals
    )  # doing it here just to get the interruptions

    stats = {}
    stats["Dauer"] = "{:02d}:{:02d}".format(*divmod(len(sp_vals), 60)) + " h"
    stats["Tief"] = "{:02d}:{:02d}".format(*divmod(sp_vals.count(1), 60)) + " h"
    stats["Leicht"] = "{:02d}:{:02d}".format(*divmod(sp_vals.count(2), 60)) + " h"
    stats["REM"] = "{:02d}:{:02d}".format(*divmod(sp_vals.count(3), 60)) + " h"
    # don't include first and last wake phase (going to sleep, waking up)
    # TODO replace interruptions with "wake time"
    stats["Unterbrechungen"] = sp_sequence[1:-1].count(4)
    stats["Mittlere HF"] = str(int(np.nanmedian(heart_rates))) + " bpm"
    stats["Mittlere AF"] = str((np.nanmedian(resp_rates))) + " bpm"

    # TODO return plot object and sleep data object instead of many lists
    return (
        sp_vals,
        stats,
        hr_averaged_plot,
        rmssd_averaged_plot,
        rr_averaged_plot,
        mvt_averaged_plot,
        hr_reference_plot,
        rmssd_reference_plot,
        rr_reference_plot,
        mvt_reference_plot,
    )


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

    fig = plt.figure()
    fig.set_size_inches(11, 10)
    fig.suptitle(f"Sleep Analysis using BCG", fontsize=16)

    # create layout
    gs = gridspec.GridSpec(nrows=11, ncols=4)

    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2:4, :2])
    ax3 = fig.add_subplot(gs[2:4, 2:])
    ax4 = fig.add_subplot(gs[4:6, 0:2])
    ax5 = fig.add_subplot(gs[4:6, 2:])
    ax6 = fig.add_subplot(gs[6:8, :2])
    ax7 = fig.add_subplot(gs[6:8, 2:])
    ax8 = fig.add_subplot(gs[-3:, :])

    # set titles
    ax1.set_title(f"Raw Sensor Data, Sampling Frequency: {fs_sm:.2f} Hz")
    ax2.set_title("Section of Filtered Respiratory Signal")
    ax3.set_title("Section of Filtered Heartrate Signal")
    ax4.set_title("Average Respiratory Rate per Minute")
    ax5.set_title("Average Heart Rate per Minute")
    ax6.set_title("Average Heart Rate Variability per Minute")
    ax7.set_title("Average Movement per Minute")
    ax8.set_title("Sleep Phases")

    # transform sleep phases to sequence of phases with corresponding length
    # for better Plot quality
    sp_sequence, sp_lengths, tick_x_pos = group_list(sleep_phases)

    # set colors for sleep phases
    cc = []
    for val in sp_sequence:
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
        rrdata_filt[int(len(rrdata_filt) * 0.2) : int(len(rrdata_filt) * 0.2) + 2575],
        color="#008000",
    )
    ax3.plot(
        # this only works for fixed sample rate of 85.2 Hz, good enough for now
        np.linspace(0, 2575 / fs_sm, 2575),
        # TODO make bounds non static
        hrdata_filt[int(len(hrdata_filt) * 0.2) : int(len(hrdata_filt) * 0.2) + 2575],
        color="#F05C3C",
    )
    ax4.bar(rr_section_timestamps, rr_sections, color="#C7E4BD", width=1)
    ax4.plot(rr_section_timestamps, rr_sections, color="#008000")
    ax5.bar(hr_section_timestamps, hr_sections, color="#F8C8BE", width=1)
    ax5.plot(hr_section_timestamps, hr_sections, color="#F05C3C")
    ax6.bar(hr_section_timestamps, hrv_sections, color="#FFCFAA", width=1)
    ax6.plot(hr_section_timestamps, hrv_sections, color="#E1701A")
    ax7.bar(rr_section_timestamps, mvt_sections, color="#E7C9FC", width=1)
    ax7.plot(rr_section_timestamps, mvt_sections, color="#9551C4")
    ax8.bar(tick_x_pos, sp_sequence, color=cc, width=sp_lengths)

    # set ticks and labels
    # ax1.set_xticks(np.arange(0, raw_timestamps[-1] / 60_000_000))
    # ax8.set_xticks(np.arange(0, len(sleep_phases)))
    ax8.set_yticks([1.0, 2.0, 3.0, 4.0], ("Deep", "Light", "REM", "Awake"))
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
    ax7.set_ylabel("Magnitude")

    # Create Textbox for sleep Stats
    props = dict(boxstyle="round", facecolor="white")
    stats_str = ", ".join("{}: {}".format(k, v) for k, v in sleep_stats.items())
    # place the text box
    ax8.text(
        0.5,
        -0.4,
        stats_str,
        transform=ax8.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=props,
    )
    # today = str(datetime.date.today())
    # plt.savefig(f"Plot_{today}.svg", format="svg")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # read the csv file into a pandas dataframe
    print("Reading Data from File...")
    FILE = "Sensordata\RawData30112021.csv"
    # name = "29112021_sensor_data"
    data = pd.read_csv(FILE, names=["timestamp", "value"], delimiter=",")
    print("Complete!")

    # calculate smoothed dataset and backfill missing data from phaseshift
    data["smoothed_ts"] = data["timestamp"].rolling(5, win_type="hanning").mean()
    data["smoothed_v"] = data["value"].rolling(5, win_type="hanning").mean()
    data["smoothed_ts"] = data["smoothed_ts"].bfill()
    data["smoothed_v"] = data["smoothed_v"].bfill()
    fs_sm = get_sampling_freq(data["smoothed_ts"].to_numpy())

    # filter (and upsample for HR) the smoothed datasets
    # TODO separate the filter output and writing to dataframe. Slows down the program
    print("Filtering Signal...")
    hr_filt = filter_hr(data["smoothed_v"], fs_sm)
    resampled_hr_filt = resample(hr_filt, len(hr_filt) * 4)
    rr_filt = filter_rr(data["smoothed_v"], fs_sm)
    print("Filtering done!")
    # write to dataframe
    data["filtered_hr"] = hr_filt
    data["filtered_rr"] = rr_filt

    # Calculate Heartrates, Respiratory Rates, HRV and Movement
    print("Calculating Heart Rates + Heartrate Variability...")
    # ignore warnings raised by heartpy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heart_rates, hrv, hr_timecodes, fail_rate = hr_hrv(
            resampled_hr_filt, (fs_sm * 4)
        )
    # remove outliers > +-5 sigma, converts to np.array
    heart_rates = reject_outliers(heart_rates, m=5)
    # data for plot to visualize failed HR calc
    hr_with_nans = np.copy(heart_rates)
    hr_only_nans = np.zeros(len(heart_rates))
    # interpolate to fill gaps, https://stackoverflow.com/a/6520696
    nans_hr, x_hr = np.isnan(heart_rates), lambda z: z.nonzero()[0]
    heart_rates[nans_hr] = np.round(
        np.interp(x_hr(nans_hr), x_hr(~nans_hr), heart_rates[~nans_hr]), 0
    )
    # plot all the HR calc stuff
    hr_only_nans[nans_hr] = 150
    sns.set(rc={"figure.figsize": (7, 4)})
    sns.set_style("whitegrid")

    plt.plot(
        heart_rates, color="grey", alpha=0.8, label="HR Interpolation", linestyle="-."
    )
    plt.plot(
        hr_with_nans,
        color="#F05C3C",
        alpha=1,
        linewidth=1.5,
        label="Heart Rate Calculated",
    )
    plt.bar(
        hr_timecodes,
        hr_only_nans,
        width=1,
        color="skyblue",
        edgecolor="none",
        alpha=0.5,
        label=f"HR Failure Rate: {fail_rate * 100:.2f} %",
    )
    plt.ylim(top=max(heart_rates) + 20)
    plt.ylabel("Herzfrequenz [bpm]")
    plt.xlabel("Zeit [Minuten]")
    plt.legend(loc="upper right")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.show()
    # needs to be np array
    hrv = np.array(hrv)
    nans_hrv, x_hrv = np.isnan(hrv), lambda z: z.nonzero()[0]
    hrv[nans_hrv] = np.round(
        np.interp(x_hrv(nans_hrv), x_hrv(~nans_hrv), hrv[~nans_hrv]), 0
    )
    print("Heartrates + Heartrate Variability done!")
    print(f"HR Failure Rate: {fail_rate * 100:.2f} %")
    # print("Calculating Respiratory Rates + Movement...")
    # resp_rates, movement, rr_timecodes = rr_mvt(rr_filt, fs_sm)
    # # remove outliers > +-5 sigma, converts to np.array
    # resp_rates = reject_outliers(resp_rates, m=5)
    # # interpolate to fill gaps, https://stackoverflow.com/a/6520696
    # nans_rr, x_rr = np.isnan(resp_rates), lambda z: z.nonzero()[0]
    # resp_rates[nans_rr] = np.round(
    #     np.interp(x_rr(nans_rr), x_rr(~nans_rr), resp_rates[~nans_rr]), 0
    # )
    # print("Respiratory Rates + Movement done!")

    # # Determine Sleep Phases
    # print("Sleep Analysis...")
    # (
    #     sp_values,
    #     sp_stats,
    #     hr_avg_plot,
    #     rmssd_avg_plot,
    #     rr_avg_plot,
    #     mvt_avg_plot,
    #     hr_ref_plot,
    #     rmssd_ref_plot,
    #     rr_ref_plot,
    #     mvt_ref_plotv,
    # ) = sleep_phases(heart_rates, hrv, resp_rates, movement)
    # print("Sleep Analysis done!")
    # print(sp_stats)

    # # plot the data of sleep phase calculation
    # plt.rcParams["figure.figsize"] = [11, 5]
    # plt.suptitle("Parameter Classification against Reference", fontsize=16)
    # plt.subplot(221, title="Heart Rate Classification Data", ylabel="HR [bpm]")
    # plt.plot(hr_avg_plot, label="HR_avg_sensor")
    # plt.plot(hr_ref_plot, label="HR_ref")
    # plt.legend()
    # plt.subplot(222, title="RMSSD Classification Data", ylabel="RMSSD [ms]")
    # plt.plot(rmssd_avg_plot, label="RMSSD_avg_sensor")
    # plt.plot(rmssd_ref_plot, label="RMSSD_ref")
    # plt.legend()
    # plt.subplot(223, title="Respiratory Rate Classification Data", ylabel="RR [bpm]")
    # plt.plot(rr_avg_plot, label="RR_avg_sensor")
    # plt.plot(rr_ref_plot, label="RR_ref")
    # plt.legend()
    # plt.subplot(224, title="Movement Classification Data", ylabel="Magnitude")
    # plt.plot(mvt_avg_plot, label="MVT_avg_sensor")
    # plt.plot(mvt_ref_plotv, label="MVT_ref")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # # write file for evaluation
    # # headings = {
    # #     "hr_sensor": list(map(int, heart_rates)),
    # #     "rr_sensor": list(map(int, resp_rates)),
    # #     "ss_sensor": list(map(int, sp_values)),
    # # }
    # # df = pd.DataFrame(headings)
    # # print(df)
    # # df.to_csv("Sensordata/clean/" + name, encoding="utf-8", index=False)

    # # Create the Plot
    # print("Plotting...")

    # plot_dash(
    #     data["smoothed_v"].to_numpy(),
    #     hr_filt,
    #     rr_filt,
    #     heart_rates,
    #     hrv,
    #     resp_rates,
    #     movement,
    #     data["smoothed_ts"].to_numpy(),
    #     hr_timecodes,
    #     rr_timecodes,
    #     sp_values,
    #     sp_stats,
    # )
