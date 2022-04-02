import pandas as pd
import matplotlib.pyplot as plt


def convert_sleep_phase(p):
    if p == 0:
        return 4
    elif p == 1:
        return 2
    elif p == 2:
        return 1
    else:
        return 3


def run(start, stop, name):
    FILE = "Withings_Data/raw_bed_hr.csv"
    withings_data_hr = pd.read_csv(FILE, delimiter=",")
    print(withings_data_hr)
    withings_data_hr["start"] = pd.to_datetime(withings_data_hr["start"])
    withings_data_hr.sort_values(by="start", inplace=True)
    withings_data_hr.drop_duplicates(inplace=True)
    withings_data_hr.reset_index(inplace=True)
    print(withings_data_hr)
    # put all values in one list
    y = eval(withings_data_hr["value"][start])
    for i in range(start + 1, stop):
        y += eval(withings_data_hr["value"][i])  # add list to previous

    FILE2 = "Withings_Data/raw_bed_respiratory-rate.csv"
    withings_data_rr = pd.read_csv(FILE2, delimiter=",")
    print(withings_data_rr)
    withings_data_rr["start"] = pd.to_datetime(withings_data_rr["start"])
    withings_data_rr.sort_values(by="start", inplace=True)
    withings_data_rr.drop_duplicates(inplace=True)
    withings_data_rr.reset_index(inplace=True)
    print(withings_data_rr)
    # put all values in one list
    y1 = eval(withings_data_rr["value"][start])
    for i in range(start + 1, stop):
        y1 += eval(withings_data_rr["value"][i])  # add list to previous

    FILE3 = "Withings_Data/raw_bed_sleep-state.csv"
    withings_data_ss = pd.read_csv(FILE3, delimiter=",")
    print(withings_data_ss)
    withings_data_ss["start"] = pd.to_datetime(withings_data_ss["start"])
    withings_data_ss.sort_values(by="start", inplace=True)
    withings_data_ss.drop_duplicates(inplace=True)
    withings_data_ss.reset_index(inplace=True)
    print(withings_data_ss)
    # put all values in one list
    y2 = eval(withings_data_ss["value"][start])
    for i in range(start + 1, stop):
        y2 += eval(withings_data_ss["value"][i])  # add list to previous
    # reassign sleep phases to correct numbers
    y2 = list(map(convert_sleep_phase, y2))

    dict = {"hr_withings": y, "rr_withings": y1, "ss_withings": y2}
    df = pd.DataFrame(dict)
    print(df)
    df.to_csv("Withings_Data/clean/" + name, encoding="utf-8", index=False)

    plt.subplot(211)
    plt.title("{:02d}:{:02d}".format(*divmod(len(y), 60)) + " h")
    plt.plot(y1)
    plt.plot(y)
    plt.subplot(212)
    plt.plot(y2)
    plt.show()
    withings_data_hr.to_csv(
        "Withings_Data/clean_hr",
        encoding="utf-8",
        columns=["start", "value"],
    )
    withings_data_rr.to_csv(
        "Withings_Data/clean_rr",
        encoding="utf-8",
        columns=["start", "value"],
    )
    withings_data_ss.to_csv(
        "Withings_Data/clean_ss",
        encoding="utf-8",
        columns=["start", "value"],
    )


start = 47
stop = 51
name = "28112021_withings_data"
run(start, stop, name)
