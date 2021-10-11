'''Reads bytes from Serial and decodes to utf-8, appends a timestamp and can write to CSV'''

from serial import Serial
from datetime import datetime


def readSerial(ser: Serial):
    """Reads Arduino 'Serial.println()' bytes from a Serial Port, converts to string and includes a timestamp"""
    if ser.in_waiting > 0:
        raw_pres_val = ser.readline()
        raw_pres_val = raw_pres_val.decode(encoding="utf8")
        raw_pres_val = raw_pres_val.replace("\r\n", "")
        timestamp = datetime.now()
        return timestamp, int(float(raw_pres_val))


def writeSerialToCSV(filepath, data: tuple):
    """Converts tuple to String and appends data to a CSV file"""
    # convert to string, milliseconds have 4 decimal places

    # TODO Format the CSV into seperate collums for Y, m, d, H, M etc.
    ts_str = data[0].strftime("%d.%m.%Y %H:%M:%S.%f")[:-2]
    raw_val_str = str(data[1])
    data_str = ts_str + ", " + raw_val_str + "\n"
    f = open(filepath, "a")
    f.write(data_str)
    f.close()


def select_COM_Port():
    """Reads a String from console"""
    COM_PORT = input("Specify COM Port (e.g. 'COM1'): ")
    return COM_PORT


def createSerial(portname: str):
    """Creates a serial port at the given port name"""
    ser = Serial(portname, 57600, timeout=0.01)
    return ser
