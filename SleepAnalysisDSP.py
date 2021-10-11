'''Reads Raw Sensor Data from CSV and performs a Sleep Analysis. Output parameters are: HeartR, RespR, MvtR, SleepPhase'''
import numpy as np

def normalize(data):
    normalized_data = data
    #TODO write normalize function
    return normalized_data

def HeartR(data):
    #TODO write heart rate calculation
    hr_vals = np.empty()
    # go through the data and calculate HR for every Minute
    return hr_vals

def RespR(data):
    #TODO write respiratory rate calculation
    rr_vals = np.empty()
    # go through the data and calculate RR for every Minute
    return rr_vals

def MvtR(data):
    #TODO write movement rate calculation
    mr_vals = np.empty()
    # go through the data and calculate MR for every Minute

    # Ansatz: Varianz Min Max berechnen und in den Kontext der Nacht setzen

    return mr_vals

def SleepPhase():
    #TODO write sleep Phase calculation
    # Ansatz: Parameter Min/Max bestimmen (für alle 3) und dann niedrigen und Hohen Bereich festlegen (einfach zweigeteilt)
    sp_vals = np.empty()
    # go through the data and calculate SP for every Minute
    return sp_vals