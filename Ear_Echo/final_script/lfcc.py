# import librosa
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils , ploter
from scipy.fft import fft,fftfreq,ifft
from scipy import signal
from scipy.special import rel_entr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import librosa
import librosa.display
import scipy
import matplotlib.pyplot as plt

F_min = 17000 # 17KHz
F_max = 23000 # 23KHz
sampling_rate = 100000 # 100KHz
Fs = 100000  # 100KHz
FMCW_duration = 0.08
silent_duration = 0.02

# Preprocess the data to obtain a segment
def pre_process(data_file_name , Inf_signal, plot=False):

    plot_name = data_file_name[8:]
    Amp_thresh , time_shift =0.0013, 0.2

    A_data , B_data, length = utils.get_data(data_file_name)
    
    Found = True
    c = 0
    while(Found):
        try:
            print(data_file_name)
            # print("Hi")
            ind = utils.segment_chirp(A_data,Fs,Amp_thresh,time_shift)
            # print(ind)
            Chirp_signal_A = ([A_data])[0][ind[0]:ind[1]]
            Chirp_signal_B = ([B_data])[0][ind[0]:ind[1]]

            Pulse_A = utils.Additionally_average(Chirp_signal_A, Fs)
            Pulse_B = utils.Additionally_average(Chirp_signal_B , Fs)

            Pulse_A = utils.butter_bandpass_filter(Pulse_A, F_min, F_max, Fs,order=9)
            Pulse_B = utils.butter_bandpass_filter(Pulse_B, F_min, F_max, Fs,order=9)

            print(abs(np.dot(Inf_signal,Pulse_A)))
            # To see how closely the signal matches the interference signal. 
            if (abs(np.dot(Inf_signal,Pulse_A))>12000): 
                Found = False

            else:
                time_shift = time_shift + 0.1
                if time_shift>1.0 : 
                    time_shift = 0.2
                    c=c+1
                
        except:
                time_shift = time_shift + 0.05
                if time_shift>1.0 : 
                    time_shift = 0.2
                    c=c+1
                    
        if c>10 : Amp_thresh= 0.0016
        elif c>15 : Amp_thresh= 0.0011
            

    if (plot):
        ploter.plot_wave(Pulse_A,Pulse_B,plot_name)


    return (Pulse_A,Pulse_B)


# #########################################################################
# Extracting the waveform of the reference signal used in segmentation matching
ref_A0d , ref_B0d, length = utils.get_data("Saadia\Data\Tes0\Ref")
ind = utils.segment_chirp(ref_A0d,Fs,thresh=0.0013,delay=0.4)
ref_A0 = ([ref_A0d])[0][ind[0]:ind[1]]
ref_B0 = ([ref_B0d])[0][ind[0]:ind[1]]

# filter the segment 
Base_refA = utils.butter_bandpass_filter(utils.Additionally_average(ref_A0,Fs), F_min, F_max, Fs,order=9)
Base_refB = utils.butter_bandpass_filter(utils.Additionally_average(ref_B0,Fs), F_min, F_max, Fs,order=9)


########################################################################## LFCC code ####################################################################

for i in range (0, 1):
    Tes_no = str(i)

    # Relax_A,Relax_B = pre_process("Saadia\Data\Tes"+Tes_no+"\Close",Base_refA,plot=True )
    Relax_A,Relax_B = pre_process("Saadia\lfcc\Data\Tes"+Tes_no+"\Close",Base_refA,plot=False )
    print("########################################",len(Relax_A))

    # OpM_A,OpM_B = pre_process("Saadia\Data\Tes"+Tes_no+"\Open",Base_refA,plot=True )
    OpM_A,OpM_B = pre_process("Saadia\lfcc\Data\Tes"+Tes_no+"\Open",Base_refA,plot=False )

    # PullL_A,PullL_B = pre_process("Saadia\Data\Tes"+Tes_no+"\PullL",Base_refA,plot=True )
    PullL_A,PullL_B = pre_process("Saadia\lfcc\Data\Tes"+Tes_no+"\PullL",Base_refA,plot=False )

    # PullR_A , PullR_B = pre_process("Saadia\Data\Tes"+Tes_no+"\PullR",Base_refA,plot=True )
    PullR_A , PullR_B = pre_process("Saadia\lfcc\Data\Tes"+Tes_no+"\PullR",Base_refA,plot=False )

    # Plot the received echo signal
    wave_matrix = np.array([[Relax_A ,OpM_A,PullL_A,PullR_A],
                            [Relax_B,OpM_B,PullL_B,PullR_B]])


    # file_path_name = "Saadia\lfcc\Results\Waves "+Tes_no+".png"
    # ploter.plot_wave_list(Tes_no , wave_matrix,file_path_name)
    
    # Get the variable names for the plot to be used as the key
    l= []
    for i in range(1, 25):
        l.append("Frame"+str(i))

    LFCC_Close_L, LFCC_Close_R  = utils.lfcc(y_L=Relax_A, y_R= Relax_B, sr=100000, n_lfcc=30, dct_type=2)
    ploter.plot_lfcc(LFCC_Close_L, LFCC_Close_R, "Relax", l)

    LFCC_OpM_L, LFCC_OpM_R = utils.lfcc(y_L= OpM_A, y_R= OpM_B, sr=100000, n_lfcc=30, dct_type=2)
    ploter.plot_lfcc(LFCC_OpM_L, LFCC_OpM_R, "Mouth open", l)

    LFCC_PullL_L, LFCC_PullL_R = utils.lfcc(y_L= PullL_A, y_R= PullL_B, sr=100000, n_lfcc=30, dct_type=2)
    ploter.plot_lfcc(LFCC_PullL_L, LFCC_PullL_R, "Pull lip left", l)

    LFCC_PullR_L, LFCC_PullR_R = utils.lfcc(y_L= PullR_A, y_R= PullR_B, sr=100000, n_lfcc=30, dct_type=2)
    ploter.plot_lfcc(LFCC_PullR_L, LFCC_PullR_R, "Pull lip right", l)


    TF_Matrix = np.array([[np.concatenate(LFCC_Close_L) ,np.concatenate(LFCC_OpM_L),np.concatenate(LFCC_PullL_L),np.concatenate(LFCC_PullR_L)],
                            [np.concatenate(LFCC_Close_R),np.concatenate(LFCC_OpM_R),np.concatenate(LFCC_PullL_R),np.concatenate(LFCC_PullR_R)]])




    # filepath = 'Saadia/lfcc/TF Data/TF'+Tes_no+'.csv' 


    # # print(new_TF_mat)
    # utils.Dump_CSV(filepath,TF_Matrix)



