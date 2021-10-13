"""Reads bytes from Serial and decodes to utf-8, appends a timestamp in microseconds and can write to CSV. Tare is provided"""

from serial import Serial
import time
import numpy as np


def readSerial(ser: Serial):
    """Reads Arduino 'Serial.println()' bytes from a Serial Port, converts to float and includes a timestamp in microseconds"""
    if ser.in_waiting > 0:
        raw_pres_val = ser.readline()
        raw_pres_val = raw_pres_val.decode(encoding="utf8")
        raw_pres_val = raw_pres_val.replace("\r\n", "")
        timestamp = round(time.time() * 1000000)
        return timestamp, float(raw_pres_val)


def tare(ser: Serial, numS: int):
    """Returns the average of numS number of read values"""
    buf = np.empty(numS)
    ptr = 0
    while ptr < numS:  # read until number of samples is reached
        val = readSerial(ser)
        if val:
            buf[ptr] = val[1]
            ptr += 1
    return sum(buf) / len(buf)  # return average value


def writeSerialToCSV(filepath: str, data: tuple, starttime: int):
    """Writes tuple of timestamp and value to String and appends data to a CSV file. Timestamps are set to zero at the start"""
    # convert to string
    str_towrite = str(data[0] - starttime) + ", " + str(data[1]) + "\n"
    f = open(filepath, "a")
    f.write(str_towrite)
    f.close()


def select_COM_Port():
    """Reads a String from console"""
    COM_PORT = input("Specify COM Port (e.g. 'COM1'): ")
    return COM_PORT


def createSerial(portname: str):
    """Creates a serial port at the given port name"""
    ser = Serial(portname, 57600, timeout=0.01)
    return ser
