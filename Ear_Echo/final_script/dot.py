import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils , ploter
from scipy.fft import fft,fftfreq,ifft
from scipy import signal
from scipy.special import rel_entr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import librosa
import librosa.display

F_min = 17000 # 17KHz
F_max = 23000 # 23KHz
sampling_rate = 100000 # 100KHz
Fs = 100000  # 100KHz
FMCW_duration = 0.08
silent_duration = 0.02

# Reference signal
ref_A0d , ref_B0d = utils.get_data("Saadia\Data\Tes0\Ref")
ind = utils.segment_chirp(ref_A0d,Fs,thresh=0.0013,delay=0.4)
ref_A0 = ([ref_A0d])[0][ind[0]:ind[1]]
ref_B0 = ([ref_B0d])[0][ind[0]:ind[1]]

Base_refA = utils.butter_bandpass_filter(utils.Additionally_average(ref_A0,Fs), F_min, F_max, Fs,order=9)
Base_refB = utils.butter_bandpass_filter(utils.Additionally_average(ref_B0,Fs), F_min, F_max, Fs,order=9)

# PullR
Amp_thresh , time_shift =0.0013, 0.2
A_data , B_data = utils.get_data("Saadia\Data\Tes0\Open")

ind = utils.segment_chirp(A_data,Fs,Amp_thresh,time_shift)
Chirp_signal_A = ([A_data])[0][ind[0]:ind[1]]
Chirp_signal_B = ([B_data])[0][ind[0]:ind[1]]

Pulse_A = utils.Additionally_average(Chirp_signal_A, Fs)
Pulse_B = utils.Additionally_average(Chirp_signal_B , Fs)

Pulse_A = utils.butter_bandpass_filter(Pulse_A, F_min, F_max, Fs,order=9)
Pulse_B = utils.butter_bandpass_filter(Pulse_B, F_min, F_max, Fs,order=9)

print(Pulse_A, Base_refA)

print((np.dot(Base_refA,Pulse_A)))