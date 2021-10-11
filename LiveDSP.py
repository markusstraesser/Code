'''Real time signal processing for serial data: Filtering and calculation of HeartR, RespR, MvtR, SleepPhase'''

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import heartpy as hp

def avg(val):
    measures = []
    
