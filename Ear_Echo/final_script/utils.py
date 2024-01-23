import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,fftfreq,ifft
# from scipy import signal
import scipy.signal as signal
from scipy.special import rel_entr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import csv
from sklearn.decomposition import PCA
import utils , ploter
from scipy import signal
import scipy
import librosa
import librosa.display


F_min = 17000 # 17KHz
F_max = 23000 # 23KHz
Fs = 100000  # 100KHz
FMCW_duration = 0.08 
silent_duration = 0.02
N = 12000
T = 1/Fs

# Parameters LFCC
n_fft= 2048
sr = Fs  # Sample rate (adjust according to your audio data)
hop_length = 512  # Hop length for STFT
n_mels = 40  # Number of Mel filter banks
n_ceps = 13  # Number of cepstral coefficients to keep (usually 12-13)

def segment_chirp(recieved_wave,fs, thresh=0.0015, delay= 0.8):
    s = normalize([recieved_wave])[0][int(delay*fs):int(2*fs)]

    chirp_confirm=False
    P=0
    while(chirp_confirm==False):
        key = np.where(s>thresh)[0][0]

        curr = key + int(0.12*fs)

        if (s[curr]>thresh):
            curr = curr+int(0.12*fs)
            if (s[curr]>thresh):
                curr = curr+int(0.12*fs)
                if (s[curr]<thresh):
                    break
                    chirp_confirm=True
                    
        P = P + curr
        s = s[curr:]
        
    P = P + key +  int(delay*fs)-int(0.02*fs)
    return( P,P+int(0.46*fs)) 


def segment_chirp_cont(recieved_wave, fs, thresh=0.0015, delay= 0.8):

    # # Number of samples in normalized_toneN = SAMPLE_RATE * DURATIONyf = fft(normalized_tone)xf = fftfreq(N, 1 / SAMPLE_RATE)plt.plot(xf, np.abs(yf))plt.show()
    s = normalize([recieved_wave])[0][int(delay*fs):int(2*fs)]
    # # Duration in seconds
    signal_duration= 0.003
    window_size= int(signal_duration*fs)

    max_frequency= 0
    max_index= 0
    start= 0
    end= start+ window_size
    while(not(int(max_frequency)>= 17000 and int(max_frequency)<= 18000)
          and not(s[max_index] >thresh and s[max_index + int(0.12*fs)] >thresh and s[max_index + int(0.12*fs)] <thresh)):
        #   and not(s[max_index] >thresh)):
          
        frequencies= []
        # get a window of the signal
        w= s[start:end]
        F_w= fft(w)
        frequencies= fftfreq(window_size, 1/fs)
        # Plot the spectrum
        # plt.plot(abs(frequencies), abs(F_w), 'k')
        # plt.title("Frequencies of each sample")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("FFT values")
        # plt.show()
        index= abs(F_w).argmax(axis=0)
        max_frequency= frequencies[index]
        max_index= index+start
        index_17= np.where(np.isclose(frequencies, 17000))
        # max_index= index_17[0][0]+start

        # print(int(max_frequency))
        start= end
        end= start + window_size

    
    # print("max_index:",max_index)
    # # P= max_index -int(0.0207*fs)
    # P= max_index -int(0.0185*fs)   
    P= max_index + int(delay*fs) - int(0.020*fs)

    # print("max frequency: ", max_frequency) 
    return(P, P+int(0.12*fs))   



def minmax_normalize(data_set):
    data_normalized = (data_set-np.min(data_set))/(np.max(data_set)-np.min(data_set))
    return data_normalized

def standard_norm(data_set):
    data_normalized= (data_set- np.mean(data_set))/(np.std(data_set) + 0.0000001)
    return data_normalized

def standardize(data_set):
    data_normalize = (data_set-np.mean(data_set))/np.std(data_set)
    return data_normalize

def butter_bandpass(lowcut, highcut, fs, order=9):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=9):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs , order=4):
    nyq_freq = fs/2
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y  

def extract_peak_frequency(data, sampling_rate):
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    
    peak_coefficient = np.argmax(np.abs(fft_data))
    peak_freq = freqs[peak_coefficient]
    
    return abs(peak_freq * sampling_rate)

def Phase_lag_Hilbert(x1,x2,sampling_rate):
    
    x1h = signal.hilbert(x1)
    x2h = signal.hilbert(x2)
    omega = (extract_peak_frequency(x1,sampling_rate) + extract_peak_frequency(x2,sampling_rate))/2
             
    c = np.inner( x1h, np.conj(x2h) ) / np.sqrt( np.inner(x1h,np.conj(x1h)) * np.inner(x2h,np.conj(x2h)) )
    phase_diff = np.angle(c)/(np.pi*2*omega)
    return(-phase_diff * sampling_rate) # return the delayed number of sample points

def read_data(str_file_name):
    df = pd.read_csv(str_file_name)
    df=df.rename(columns={'Time':'time', 'Channel A':'ChnA', 'Channel B':'ChnB'})
    t = np.array(df.time[1:]).astype(float)
    Chn_A = np.array(df.ChnA[1:]).astype(float)
    Chn_B = np.array(df.ChnB[1:]).astype(float)
   
    return(t/1000 , Chn_A, Chn_B, len(df.index))

def get_data(filename):
    dA_1,dB_1 = np.array([]),np.array([])
    for i in range(1,3):
        s = str(i)
        file_name = filename + ".csv"
        time,A,B, len = read_data(file_name)
        dA_1,dB_1 = np.concatenate((dA_1,A)),np.concatenate((dB_1,B))
    return(dA_1,dB_1, len)


def Additionally_average(data_set,fs):
    fs = int(fs)
    averaged_data = data_set[: int(0.120*fs)]

    for i in range(1,3):
     
        averaged_data= averaged_data + data_set[int((0.120*i)*fs) : int((0.120*(i+1))*fs)]
    return(averaged_data.astype(float)/3)

def filtering(array,xf):
    a = np.zeros(len(array))
    for i in range(len(xf)):
        if ((xf[i])>F_min and (xf[i])<F_max):
            a[i]=(array[i]) 

    return (a)

def Dump_CSV(filepath, array):
    # Write arrays to a CSV file
    

    # [[TF_Close_L,TF_OpM_L,TF_PullL_L,TF_PullR_L,TF_Eye_L],[TF_Close_R,TF_OpM_R,TF_PullL_R,TF_PullR_R,TF_Eye_R]] = array
    [[TF_Close_L,TF_OpM_L,TF_PullL_L,TF_PullR_L],[TF_Close_R,TF_OpM_R,TF_PullL_R,TF_PullR_R]] = array


    # combinedTF_array = np.column_stack((TF_Close_L,TF_OpM_L,  
    #                                         TF_PullL_L, TF_PullR_L,        
    #                                         TF_Eye_L,TF_Close_R,
    #                                         TF_OpM_R, TF_PullL_R,
    #                                         TF_PullR_R,TF_Eye_R))

    combinedTF_array = np.column_stack((TF_Close_L,TF_OpM_L,  
                                            TF_PullL_L, TF_PullR_L ,TF_Close_R,
                                            TF_OpM_R, TF_PullL_R,
                                            TF_PullR_R))

    # print(combinedTF_array)
    with open(filepath, 'w', newline='') as csvfile:
        
        try:
            writer = csv.writer(csvfile)
            
            # Write column headers (optional)
            # writer.writerow(['RelaxL', 'OpenMouthL', 'lipLeftL', 'lipRightL','EyeL','RelaxR', 'OpenMouthR', 'lipLeftR', 'lipRightR','EyeR'])
            writer.writerow(['RelaxL', 'OpenMouthL', 'lipLeftL', 'lipRightL','RelaxR', 'OpenMouthR', 'lipLeftR', 'lipRightR'])
            
            # Write array data
            writer.writerows(combinedTF_array)

        except:
            return False

    # FOR the Transfer Functions # Extrancting important frequency domain data


    return

def Dump_CSV_cont(filepath, array):
    # Write arrays to a CSV file

    # Check how many chirps in the array
    num_chirps= len(array[0])
    row_vals= list()

    # Get all the chirp data of both left and right ears
    for i in range (num_chirps):
        row_vals.insert(i, list(array[0][i]) + list(array[1][i]))
        

    # Stack the arrays as rows
    combinedTF_array = np.row_stack(row_vals)

    with open(filepath, 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        
        # Write array data
        writer.writerows(combinedTF_array)

    return

# method to get the PCA component
def get_pcacomponents(input_data, n):
    # 0 < n_components < min(n_samples, n_features)
    n_samples= len(input_data)
    n_features= len(input_data[0])
    
    # pca_model= PCA(n_components= min(n_samples, n_features))
    pca_model= PCA(n_components= n)
    pca_model.fit(input_data)
    # print("PCA Components")
    # print(pca_model.components_)
    return pca_model

def get_features(gesture_data, pca_components):
    M= []
    for component in pca_components:
        M.append(np.dot(gesture_data, component))


    return M


# # Sample 1 and the two PCAs
# feature11= np.dot(X[0], pca_components[0])
# feature12= np.dot(X[0], pca_components[1])
# print(feature11, feature12)
# # Sample 2 and the two PCAs
# feature21= np.dot(X[1], pca_components[0])
# feature22= np.dot(X[1], pca_components[1])
# print(feature21, feature22)

# feature_matrix= np.array([[feature11, feature12], [feature21, feature22]])
    
def lin(sr, n_fft, n_filter=128, fmin=17000.0, fmax=23000.0, dtype=np.float32):
# def lin(sr, n_fft, n_filter=128, fmin=17000.0, fmax=23000.0, dtype=np.float32): ##### ------------> Try this line if the above doesn't give good accuracy

    if fmax is None:
        fmax = float(sr) / 2
    # Initialize the weights
    n_filter = int(n_filter)
    weights = np.zeros((n_filter, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of linear bands - uniformly spaced between limits
    linear_f = np.linspace(fmin, fmax, n_filter + 2)

    fdiff = np.diff(linear_f)
    ramps = np.subtract.outer(linear_f, fftfreqs)

    for i in range(n_filter):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    return weights

def linear_spec(y=None, sr=100000, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0, **kwargs):
    S = np.abs(
        librosa.core.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=window,
                          center=center,
                          pad_mode=pad_mode))**power
    filter = lin(sr=sr, n_fft=n_fft, **kwargs)
    return np.dot(filter, S)


def lfcc(y_L=None, y_R=None, sr=100000, S_L=None, S_R=None, n_lfcc=25, dct_type=2, norm='ortho', **kwargs):
    # # fft of the signals
    # yL_fft= fft(y_L)
    # yR_fft= fft(y_R)
    # refL_fft= fft(ref_L)
    # refR_fft= fft(ref_R)

    # # Tranform domain
    # # L, R= F_ref_A0/refL_fft, F_ref_B0/refR_fft

    # # Remove interference and convert all signals to one domain
    # pure_yL_fft= (yL_fft) 
    # pure_yR_fft= (yR_fft) 

    # # Inverse fft of the pure signal
    # yL= ifft(pure_yL_fft).real
    # yR= ifft(pure_yR_fft).real

    if S_L is None:
        S_L = librosa.power_to_db(linear_spec(y=y_L, sr=sr, **kwargs))

    if S_R is None:
        S_R = librosa.power_to_db(linear_spec(y=y_R, sr=sr, **kwargs))

    M_L = scipy.fftpack.dct(S_L, axis=0, type=dct_type, norm=norm)[:n_lfcc]
    print(len(scipy.fftpack.dct(S_L, axis=0, type=dct_type, norm=norm)))
    M_R = scipy.fftpack.dct(S_R, axis=0, type=dct_type, norm=norm)[:n_lfcc]

    return (M_L, M_R)
