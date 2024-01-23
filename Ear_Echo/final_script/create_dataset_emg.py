import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.fft import fft, fftfreq
import utils
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

Fs= 512

# pre process the emg data
def pre_process(data_file_name):

    data = pd.read_csv(data_file_name)
    arrL, arrR = data['5'], data['6']

    filtered_dataL, filtered_dataR= utils.butter_lowpass_filter(arrL, 10  , Fs, order=4), utils.butter_lowpass_filter(arrR, 10  , Fs, order=4)

    return filtered_dataL, filtered_dataR

# names= ['open', 'eye', 'pulll', 'pullr', 'relax']
names=['eye', 'gazeleft', 'Gazeright','Open', 'Pulll', 'Pullr', 'Relax', 'Updown']
# names=['Open']
# sampling_rate= 10000
# index= '5'

experiments=[]

for name in names:
    for i in range (1, 31):
        sig_, sigR_= pre_process("Saadia/Ensemble/emg/"+name+str(i)+".csv")
        sig, sigR= utils.minmax_normalize(sig_), utils.minmax_normalize(sigR_)
        
        ch1, ch2= pd.Series(sig), pd.Series(sigR)
        experiments.append(pd.concat([ch1, ch2],axis=0))

# Store the processed emg data in a csv file
with open("Saadia/Ensemble/"+"dataset.csv", 'w', newline='') as csvfile:
    writer= csv.writer(csvfile)
    writer.writerows(experiments)



