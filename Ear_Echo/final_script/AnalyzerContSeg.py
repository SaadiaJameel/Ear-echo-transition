import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils , ploter
from scipy.fft import fft,fftfreq,ifft
from scipy import signal
from scipy.special import rel_entr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from scipy.signal import correlate
import librosa
import librosa.display
import csv

F_min = 17000 # 17KHz
F_max = 23000 # 23KHz
sampling_rate = 100000 # 100KHz
Fs = 100000  # 100KHz
FMCW_duration = 0.08
silent_duration = 0.02

#########################################################################
# Extracting the waveform of the reference signal which is corresponding to a pre-defined ear-bud position.
# This is used to time shift and correct the phase of the obtained segment

ref_A0d , ref_B0d, length = utils.get_data("Saadia\Sequence\Data\Tes22\Ref")
ind = utils.segment_chirp(ref_A0d,Fs,thresh=0.0013,delay=0.4)
ref_A0 = ([ref_A0d])[0][ind[0]:ind[1]]
ref_B0 = ([ref_B0d])[0][ind[0]:ind[1]]

Base_refA = utils.butter_bandpass_filter(utils.Additionally_average(ref_A0,Fs), F_min, F_max, Fs,order=9)
Base_refB = utils.butter_bandpass_filter(utils.Additionally_average(ref_B0,Fs), F_min, F_max, Fs,order=9)


# The continuous sequence itself.
cont_A0_d, cont_B0_d, length= utils.get_data("Saadia\Sequence\Data\Tes40\Seq")

# Get the first chirp
ind= utils.segment_chirp_cont(cont_A0_d, Fs, thresh=0.0013, delay=0.4)
# Segment the signal
cont_A0 = ([cont_A0_d])[0][ind[0]:ind[1]]
cont_B0 = ([cont_B0_d])[0][ind[0]:ind[1]]

# Filtering
contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)

# Calculate the correlation between the segment and the reference signal to compare signals
correlation_result= correlate(contA0, Base_refA, mode='full')

time_shift= np.argmax(correlation_result) - (len(contA0) - 1)
time_shift_seconds= time_shift/Fs
print(time_shift_seconds)

# Get the corrected indices for the segment
ind= (ind[0]+int(time_shift_seconds*Fs), ind[1] + int(time_shift_seconds*Fs))


# M= [[],[]]

# A list to store the LFCC features of each segment
row_vals= list()
count= 0

# If the corrected indice exists 
if(ind[0] >= 0):

    # Segment the signal
    cont_A0 = ([cont_A0_d])[0][ind[0] : ind[1]]
    cont_B0 = ([cont_B0_d])[0][ind[0] : ind[1]]

    print(ind)

    contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
    contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)

    # Compare the signals and match them on the time axis
    correlation_result= correlate(contA0, Base_refA, mode='full')

    time_shift= np.argmax(correlation_result) - (len(contA0) - 1)
    time_shift_seconds= time_shift/Fs

    ind= (ind[0]+int(time_shift_seconds*Fs), ind[1] + int(time_shift_seconds*Fs))

    # Corrected segment
    cont_A0 = ([cont_A0_d])[0][ind[0] : ind[1]]
    cont_B0 = ([cont_B0_d])[0][ind[0] : ind[1]]

    contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
    contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)


    print("Size:", ind[0]+int((time_shift_seconds)*Fs))
    print("COUNT(init)= ", count, "         index(init)= ", ind[0])
    ploter.plot_wave(contA0, contB0, "Segment")

    # Get the LFCC features of the segmnet
    LFCC_cont_L, LFCC_cont_R  = utils.lfcc(y_L=contA0, y_R= contB0, sr=100000, n_lfcc=30, dct_type=2)
    new_col= (list(np.concatenate(LFCC_cont_L))+list(np.concatenate(LFCC_cont_R)))

    row_vals.insert(count, new_col)
    count= count + 1

    
# Get all the chirps in the file and analyze them.
while(ind[1] <= length):

    if(count != 0 and count%3 == 0):
        ind= (ind[1] + int(0.1*Fs), ind[1] + int(0.22*Fs))    # Start and end of the next chirp
    else:   
        ind= (ind[1], ind[1] + int(0.12*Fs))    # Start and end of the next chirp
    # Get the next segment
    cont_A0 = ([cont_A0_d])[0][ind[0]:ind[1]]
    cont_B0 = ([cont_B0_d])[0][ind[0]:ind[1]]
    # Average and filter
    contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
    contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)

    correlation_result= correlate(contA0, Base_refA, mode='full')

    time_shift= np.argmax(correlation_result) - (len(contA0) - 1)
    time_shift_seconds= time_shift/Fs

    ind= (ind[0]+int(time_shift_seconds*Fs), ind[1] + int(time_shift_seconds*Fs))

    # Segmented signal
    cont_A0 = ([cont_A0_d])[0][ind[0] : ind[1]]
    cont_B0 = ([cont_B0_d])[0][ind[0] : ind[1]]

    contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
    contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)

    print("COUNT= ", count, "         index(init)= ", ind[0])
    ploter.plot_wave(contA0, contB0, "Segment")

    # Get the LFCC coefficients
    LFCC_cont_L, LFCC_cont_R  = utils.lfcc(y_L=contA0, y_R= contB0, sr=100000, n_lfcc=30, dct_type=2)

    new_col= (list(np.concatenate(LFCC_cont_L))+list(np.concatenate(LFCC_cont_R)))
    row_vals.insert(count, new_col)
    count= count + 1


# Store the features of each segment in a file
filepath= "Saadia\lfcc\TF Data\cont_seg.csv"
    
with open(filepath, 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        
        # Write array data
        writer.writerows(np.row_stack(row_vals))