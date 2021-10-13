"""Writes serial data to CSV and prints live data to console"""

import time
import ArduinoSerial

if __name__ == "__main__":
    PORT = ArduinoSerial.select_COM_Port()
    # PORT = "COM4"
    SER = ArduinoSerial.createSerial(PORT)
    FILEPATH = "RawData.csv"
    # start time in microseconds
    starttime = round(time.time() * 1000000)
    while True:
        data = ArduinoSerial.readSerial(SER)
        if data:
            ArduinoSerial.writeSerialToCSV(FILEPATH, data, starttime)
            print(str(data[0] - starttime) + ": " + str(data[1]), end="\r")
