import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.fft import fft, fftfreq
import utils, ploter
from scipy.signal import spectrogram, find_peaks, hilbert
import pywt
import csv
from scipy.signal import cwt, ricker
from scipy.fft import fft,fftfreq,ifft
# from scipy import signal
import scipy.signal as signal
from scipy.special import rel_entr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import librosa
import csv
import librosa.display
from sklearn.decomposition import PCA
import ruptures as rpt
from scipy.signal import correlate

Fs= 512

# A method to pre process the emg signal data
def pre_process(data_file_name):
    data = pd.read_csv(data_file_name)
    arrL, arrR = data['5'], data['6']

    filtered_dataL, filtered_dataR= utils.butter_lowpass_filter(arrL, 10  , Fs, order=4), utils.butter_lowpass_filter(arrR, 10  , Fs, order=4)

    return filtered_dataL, filtered_dataR



# Get the PCA components for the lower facial region
Y_Lower= pd.read_csv("Saadia/Ensemble/dataset_lower_face.csv", sep=',', header=None)

print(Y_Lower.values.shape)

i= len(Y_Lower.values)-1     
pca_model_lower= utils.get_pcacomponents(Y_Lower.values, i)
pca_components_lower= pca_model_lower.components_

# Get the PCA components for the upper facial region
Y_Upper= pd.read_csv("Saadia/Ensemble/dataset_upper_face.csv", sep=',', header=None)

print(Y_Upper.values.shape)

i= len(Y_Upper.values)-1     
pca_model_upper= utils.get_pcacomponents(Y_Upper.values, i)
pca_components_upper= pca_model_upper.components_


#************************************************************************************************#

# Read the continuous EMG data
sig_, sigR_= pre_process("Saadia/Ensemble/emg/eye1"+".csv")
sig, sigR= utils.minmax_normalize(sig_), utils.minmax_normalize(sigR_)
ch1, ch2= pd.Series(sig), pd.Series(sigR)
data= pd.concat([ch1, ch2],axis=0)
input= data.to_numpy()


# Segment the continuous stream
algo = rpt.Pelt(model="l2", min_size=1024)
algo.fit(input)
result = algo.predict(pen=1)
print(result)

rpt.display(input, [], result)


# Get the time stamps to map the segments to the ear data
sampling_rate_emg= 512
t=[]
for sample_no in result:
  t.append(sample_no/sampling_rate_emg)
print(t)

# Get the samplenumber for the ear data
sampling_rate_ear= 100000
index=[]
for t in t:
  index.append(int(sampling_rate_ear*t))
print(index)


###################################### Map the emg segment to the ear data time series #########################################
F_min = 17000 # 17KHz
F_max = 23000 # 23KHz
sampling_rate = 100000 # 100KHz
Fs = 100000  # 100KHz
FMCW_duration = 0.08
silent_duration = 0.02
# Extracting the waveform of the reference signal which is corresponding to a pre-defined ear-bud position.

ref_A0d , ref_B0d, length = utils.get_data("Saadia\Sequence\Data\Tes22\Ref")
ind = utils.segment_chirp(ref_A0d,Fs,thresh=0.0013,delay=0.4)
ref_A0 = ([ref_A0d])[0][ind[0]:ind[1]]
ref_B0 = ([ref_B0d])[0][ind[0]:ind[1]]

Base_refA = utils.butter_bandpass_filter(utils.Additionally_average(ref_A0,Fs), F_min, F_max, Fs,order=9)
Base_refB = utils.butter_bandpass_filter(utils.Additionally_average(ref_B0,Fs), F_min, F_max, Fs,order=9)
###################################################

ear_l, ear_r, length= utils.get_data('Saadia/Ensemble/ear/Eye1')

# Use these indices to get the ear segments that need to be analyzed. 
for i in range(len(index)-1):

    # Get the EMG PCA features for lower and upper region for this segment
    values= input[result[i]:result[i+1]]
    # Zero pad the array until the length is 8785
    if(len(values)<8785):
        values=np.pad(values, (0, 8785-len(values)), 'constant', constant_values=0)
    else:
        values= values[:8785]
    print(len(values))

    data=[]                                     # Holds the PCA components as features
    colnames= []
    # Lower PCA
    data.append(utils.get_features(values, pca_components_lower))
    # Upper PCA
    data.append(utils.get_features(values, pca_components_upper))

    print(index[i], index[i+1])
    seg_l= ear_l[index[i]:index[i+1]]
    seg_r= ear_r[index[i]:index[i+1]]

    plt.plot(seg_l)
    plt.title("EMG referenced segmentation")
    plt.show()

    # Get the first chirp
    ind= utils.segment_chirp_cont(seg_l, Fs, thresh=0.0013, delay=0.4)
    # Segmented signal
    cont_A0 = ([seg_l])[0][ind[0]:ind[1]]
    cont_B0 = ([seg_r])[0][ind[0]:ind[1]]

    # Ready to get the transfer function
    contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
    contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)

    # Calculate the correlation between the segment and the Inference signal
    correlation_result= correlate(contA0, Base_refA, mode='full')

    time_shift= np.argmax(correlation_result) - (len(contA0) - 1)
    time_shift_seconds= time_shift/Fs
    print(time_shift_seconds)

    ind= (ind[0]+int(time_shift_seconds*Fs), ind[1] + int(time_shift_seconds*Fs))

    M= [[],[]]

    row_vals= list()                            # Holds the ear LFCC features for each segment
    count= 0

    if(ind[0] >= 0):

        # Segmented signal
        cont_A0 = ([seg_l])[0][ind[0] : ind[1]]
        cont_B0 = ([seg_r])[0][ind[0] : ind[1]]

        print(ind)

        contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
        contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)

        correlation_result= correlate(contA0, Base_refA, mode='full')

        time_shift= np.argmax(correlation_result) - (len(contA0) - 1)
        time_shift_seconds= time_shift/Fs

        ind= (ind[0]+int(time_shift_seconds*Fs), ind[1] + int(time_shift_seconds*Fs))

        # Segmented signal
        cont_A0 = ([seg_l])[0][ind[0] : ind[1]]
        cont_B0 = ([seg_r])[0][ind[0] : ind[1]]

        contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
        contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)

        ##### Get the features of the ear segment #####
        # LFCC
        LFCC_cont_L, LFCC_cont_R  = utils.lfcc(y_L=contA0, y_R= contB0, sr=100000, n_lfcc=30, dct_type=2)
        new_col= (list(np.concatenate(LFCC_cont_L))+list(np.concatenate(LFCC_cont_R)))
        row_vals.insert(count, new_col)

        print("Size:", ind[0]+int((time_shift_seconds)*Fs))
        print("COUNT(init)= ", count, "         index(init)= ", ind[0])
        ploter.plot_wave(contA0, contB0, "Segment")

    # Get all the chirps in the file and analyze them.
    while(ind[1] <= index[i+1]):

        if(count != 0 and count%3 == 0):
            ind= (ind[1] + int(0.1*Fs), ind[1] + int(0.22*Fs))    # Start and end of the next chirp
        else:   
            ind= (ind[1], ind[1] + int(0.12*Fs))    # Start and end of the next chirp
        # Get the next segment
        cont_A0 = ([seg_l])[0][ind[0]:ind[1]]
        cont_B0 = ([seg_r])[0][ind[0]:ind[1]]

        # Average and filter
        contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
        contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)

        if(len(contA0)== 12000):

            correlation_result= correlate(contA0, Base_refA, mode='full')

            time_shift= np.argmax(correlation_result) - (len(contA0) - 1)
            time_shift_seconds= time_shift/Fs

            ind= (ind[0]+int(time_shift_seconds*Fs), ind[1] + int(time_shift_seconds*Fs))

            # Segmented signal
            cont_A0 = ([seg_l])[0][ind[0] : ind[1]]
            cont_B0 = ([seg_r])[0][ind[0] : ind[1]]

            contA0= utils.butter_bandpass_filter(cont_A0, F_min, F_max, Fs, order=9)
            contB0= utils.butter_bandpass_filter(cont_B0, F_min, F_max, Fs, order=9)

            ##### Get the features of the ear segment #####
            # LFCC
            LFCC_cont_L, LFCC_cont_R  = utils.lfcc(y_L=contA0, y_R= contB0, sr=100000, n_lfcc=30, dct_type=2)
            new_col= (list(np.concatenate(LFCC_cont_L))+list(np.concatenate(LFCC_cont_R)))
            row_vals.insert(count, new_col)
           
            print("COUNT= ", count, "         index= ", ind[0], "       length= ", len(contA0), len(contB0))
            ploter.plot_wave(contA0, contB0, "Segment")

    # Write the obtained emg features
    with open("Saadia/Ensemble/cont/"+"emg_features"+str(i)+".csv", 'w', newline='') as csvfile:
        writer= csv.writer(csvfile)
        writer.writerows(data)

    # Write the obtained emg features
    with open("Saadia/Ensemble/cont/"+"ear_features"+str(i)+".csv", 'w', newline='') as csvfile:
        writer= csv.writer(csvfile)
        writer.writerows(np.row_stack(row_vals))
    
    # Handle the last segment
    i=i+2
    if(i >= len(index)-1):
       break

    
        

    








