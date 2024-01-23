import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.signal import butter
import scipy.signal as signal

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

F_min = 14000 # 16KHz
F_max = 24000 # 20KHz
sampling_rate = 192000 # 384KHz

FMCW_duration = 0.08
silent_duration = 0.02
chirp_signal = librosa.chirp(fmin=F_min,fmax=F_max,sr=sampling_rate,duration=FMCW_duration,linear=False)


tone = librosa.tone(0, sr=sampling_rate, length=int(silent_duration*sampling_rate))
# silent = np.zeros(int(silent_duration*sampling_rate))

transmition_signal = np.array([])

for i in range(3):
    transmition_signal = np.concatenate((transmition_signal ,tone,chirp_signal ,tone))
transmition_signal = np.concatenate((transmition_signal ,tone,tone,tone,tone,tone))

filtered_transmition_signal = butter_bandpass_filter(transmition_signal,F_min,F_max,fs=sampling_rate,order=9)

final = np.array([])
for j in range(50):
    final = np.concatenate((final,filtered_transmition_signal))

sf.write("TR 14-24_filtered.wav",final,sampling_rate)
sound,sr = librosa.load("TR 18-24_filtered.wav",sr=sampling_rate)
plt.figure(figsize=(15,4))
#plt.plot(time,soundl)
librosa.display.waveshow(sound,sr=sampling_rate,color="green")
#plt.xlim((0,1))

D = librosa.amplitude_to_db(np.abs(librosa.stft(sound)), ref=np.max)
plt.figure(figsize=(15,6))
img = librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=sampling_rate)                         
plt.colorbar()
plt.xlim((0,0.5))
plt.ylim((0000,40000))
plt.show()