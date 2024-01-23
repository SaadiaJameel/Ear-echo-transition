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

# 0.0013, 0.2 , 12000 Most acc for 1 Person

def pre_process(data_file_name , Inf_signal, plot=False):

    plot_name = data_file_name[8:]
    Amp_thresh , time_shift =0.0013, 0.2

    A_data , B_data, length = utils.get_data(data_file_name)
    
    # print(type(pd.DataFrame(A_data)))
    
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
            
    # Pulse_A = utils.minmax_normalize(Pulse_A )       
    # Pulse_B = utils.minmax_normalize(Pulse_B)

    if (plot):
        ploter.plot_wave(Pulse_A,Pulse_B,plot_name)
    
    # print(len(Pulse_A))

    return (Pulse_A,Pulse_B)


##########################################################################
# Extracting the wave form of the interferance signal 

Intfer_A_d,Intfer_B_d, length = utils.get_data("Dulaj\Infer")
ind = utils.segment_chirp(Intfer_A_d,Fs,thresh=0.0013,delay=0.6)
Intfer_A = ([Intfer_A_d])[0][ind[0]:ind[1]]
# print(Intfer_A.size)
Intfer_B = ([Intfer_B_d])[0][ind[0]:ind[1]]

Inf_A =    utils.butter_bandpass_filter(utils.Additionally_average(Intfer_A,Fs), F_min, F_max, Fs,order=9)
Inf_B =   utils.butter_bandpass_filter(utils.Additionally_average(Intfer_B,Fs), F_min, F_max, Fs,order=9)

# print(Inf_A.size)

#ploter.plot_wave(Inf_A , Inf_B , "Inferance")

#Inf_A , Inf_B = utils.minmax_normalize(Inf_A) , utils.minmax_normalize(Inf_B)

#########################################################################
# Extracting the waveform of the refarance signal which is corresponding to a pre-define ear-bud position.

ref_A0d , ref_B0d, length = utils.get_data("Dulaj\Data\Tes0\Ref")
ind = utils.segment_chirp(ref_A0d,Fs,thresh=0.0013,delay=0.4)
ref_A0 = ([ref_A0d])[0][ind[0]:ind[1]]
ref_B0 = ([ref_B0d])[0][ind[0]:ind[1]]

Base_refA = utils.butter_bandpass_filter(utils.Additionally_average(ref_A0,Fs), F_min, F_max, Fs,order=9)
Base_refB = utils.butter_bandpass_filter(utils.Additionally_average(ref_B0,Fs), F_min, F_max, Fs,order=9)

# print(abs(np.dot(Inf_A,Base_refA)))

#Base_refA , Base_refB = utils.minmax_normalize(Base_refA) , utils.minmax_normalize(Base_refB)

##########################################################################
# # # Continuous sequence analysis

# cont_A0_d, cont_B0_d, length= utils.get_data("Saadia\Sequence\Data\Tes0\Seq")
# # Get the first chirp
# ind= utils.segment_chirp(cont_A0_d, Fs, thresh=0.0013, delay=0.4)
# cont_A0 = ([cont_A0_d])[0][ind[0]:ind[1]]
# cont_B0 = ([cont_B0_d])[0][ind[0]:ind[1]]

# # Ready to get the transfer function
# contA0= utils.butter_bandpass_filter(utils.Additionally_average(cont_A0, Fs), F_min, F_max, Fs, order=9)
# contB0= utils.butter_bandpass_filter(utils.Additionally_average(cont_B0, Fs), F_min, F_max, Fs, order=9)

T= 1/Fs
N=len(Base_refA)
X_f = fftfreq(N,T)

print("N before:", N)

F_interferance_A , F_interferance_B = fft(Inf_A) , fft(Inf_B) # Achieving the FFT for above derived wave forms
print("After fft:", Inf_A.size)
F_ref_A0 , F_ref_B0 = fft(Base_refA) , fft(Base_refB)
# F_cont_A0, F_cont_B0= fft(contA0), fft(contB0)

print("N after:", len(F_ref_A0))

def get_TF(signalA,signalB, ref_A, ref_B ):
    
    signalA_fft , signalB_fft = fft(signalA) , fft(signalB)
    F_ref_A , F_ref_B = fft(ref_A) , fft(ref_B)

    A , B = F_ref_A0/F_ref_A , F_ref_B0/F_ref_B
    
    tfA= ((signalA_fft*A) - F_interferance_A) / F_interferance_A

    tfB = ((signalB_fft*B)  -F_interferance_B) / F_interferance_B

    # tf_A_dB = 10*np.log10(abs(tfA))
    # tf_B_dB = 10*np.log10(abs(tfB))
    print("From inside:", tfA.size)

    print(tfA)
    return(tfA,tfB)

def prepare2dump(tfxa):
    lower_cut = int(len(X_f)/2 * (F_min)/max(X_f)) + 1
    upper_cut = int(len(X_f)/2 * (F_max)/max(X_f)) + 1

    dumping_tf = utils.filtering( 10*np.log10(abs(tfxa) ) , X_f)[lower_cut : upper_cut]

    return (dumping_tf)


# # for i in range(11,12)

for i in range (2, 20):
# i= 56
    Tes_no = str(i)

    # Ref_A,Ref_B = pre_process("Saadia\Data\Tes"+Tes_no+"\Ref" ,Base_refA,plot=True )
    Ref_A,Ref_B = pre_process("Dulaj\Data\Tes"+Tes_no+"\Ref" ,Base_refA,plot=False )

    # _lax_A,Relax_B = pre_process("Saadia\Data\Tes"+Tes_no+"\Close",Base_refA,plot=True )
    Relax_A,Relax_B = pre_process("Dulaj\Data\Tes"+Tes_no+"\Close",Base_refA,plot=False )

    # OpM_A,OpM_B = pre_process("Saadia\Data\Tes"+Tes_no+"\Open",Base_refA,plot=True )
    OpM_A,OpM_B = pre_process("Dulaj\Data\Tes"+Tes_no+"\Open",Base_refA,plot=False )

    # PullL_A,PullL_B = pre_process("Saadia\Data\Tes"+Tes_no+"\PullL",Base_refA,plot=True )
    PullL_A,PullL_B = pre_process("Dulaj\Data\Tes"+Tes_no+"\PullL",Base_refA,plot=False )

    # PullR_A , PullR_B = pre_process("Saadia\Data\Tes"+Tes_no+"\PullR",Base_refA,plot=True )
    PullR_A , PullR_B = pre_process("Dulaj\Data\Tes"+Tes_no+"\PullR",Base_refA,plot=False )

    # EyeUp_A , EyeUp_B = pre_process("Saadia\Data\Tes"+Tes_no+"\Eye",Base_refA,plot=True )
    # EyeUp_A , EyeUp_B = pre_process("Saadia\Data\Tes"+Tes_no+"\Eye",Base_refA,plot=False )

    # # Reference signal of the continuous sequence
    # Ref_cont_A, Ref_cont_B= pre_process("Saadia\Sequence\Data\Tes0\Ref", Base_refA, plot=False)

    # wave_matrix = np.array([[Relax_A ,OpM_A,PullL_A,PullR_A,EyeUp_A],
    #                         [Relax_B,OpM_B,PullL_B,PullR_B,EyeUp_B]])

    wave_matrix = np.array([[Relax_A ,OpM_A,PullL_A,PullR_A],
                            [Relax_B,OpM_B,PullL_B,PullR_B]])


    file_path_name = "Dulaj\Results\Waves "+Tes_no+".png"
    ploter.plot_wave_list(Tes_no , wave_matrix,file_path_name)

    TF_Close_L , TF_Close_R = get_TF(Relax_A,Relax_B ,Ref_A,Ref_B )
    TF_OpM_L , TF_OpM_R = get_TF(OpM_A,OpM_B ,Ref_A,Ref_B )
    TF_PullL_L , TF_PullL_R = get_TF(PullL_A,PullL_B ,Ref_A,Ref_B )
    TF_PullR_L , TF_PullR_R = get_TF(PullR_A , PullR_B  ,Ref_A,Ref_B )
    # TF_Eye_L , TF_Eye_R = get_TF(EyeUp_A , EyeUp_B ,Ref_A,Ref_B )

    # # Transfer function of the continuous sequence
    # TF_cont_L, TF_cont_R= get_TF(contA0, contB0, Ref_cont_A, Ref_cont_B)

    # TF_Matrix = np.array([[TF_Close_L ,TF_OpM_L,TF_PullL_L,TF_PullR_L,TF_Eye_L],
    #                         [TF_Close_R,TF_OpM_R,TF_PullL_R,TF_PullR_R,TF_Eye_R]])

    TF_Matrix = np.array([[TF_Close_L ,TF_OpM_L,TF_PullL_L,TF_PullR_L],
                            [TF_Close_R,TF_OpM_R,TF_PullL_R,TF_PullR_R]])

    # TF_cont_Matrix= np.array([[TF_cont_L], [TF_cont_R]])
    # Plot the output (Channel Responses of each facial expressions)

    file_path_name = "Dulaj\Results\TF_"+"L "+Tes_no+".png"
    ploter.plot_tf(Tes_no, N ,T , TF_Matrix[0], file_path_name,Left = True)

    file_path_name = "Dulaj\Results\TF_"+"R "+Tes_no+".png"
    ploter.plot_tf(Tes_no, N ,T , TF_Matrix[1],file_path_name,Left = False)

    # file_path_name = "Saadia\Sequence\Results\TF_L0.png"
    # ploter.plot_tf(0, N, T, TF_cont_Matrix[0], file_path_name, Left= True)

    # file_path_name= "Saadia\Sequence\Results\TF_R0.png"
    # ploter.plot_tf(0, N, T, TF_cont_Matrix[1], file_path_name, Left = False)

    # Saving the above data in to CSV files.


    # Define cutoff frequency ranges that need to store


    filepath = 'Dulaj/TF Data/TF'+Tes_no+'.csv' 

    # new_TF_mat = np.array([[prepare2dump(TF_Close_L) ,prepare2dump(TF_OpM_L),prepare2dump(TF_PullL_L),prepare2dump(TF_PullR_L),prepare2dump(TF_Eye_L)],
    #                         [prepare2dump(TF_Close_R),prepare2dump(TF_OpM_R),prepare2dump(TF_PullL_R),prepare2dump(TF_PullR_R),prepare2dump(TF_Eye_R)]])

    new_TF_mat = np.array([[prepare2dump(TF_Close_L) ,prepare2dump(TF_OpM_L),prepare2dump(TF_PullL_L),prepare2dump(TF_PullR_L)],
                            [prepare2dump(TF_Close_R),prepare2dump(TF_OpM_R),prepare2dump(TF_PullL_R),prepare2dump(TF_PullR_R)]])


    # print(new_TF_mat)
    utils.Dump_CSV(filepath,new_TF_mat)



