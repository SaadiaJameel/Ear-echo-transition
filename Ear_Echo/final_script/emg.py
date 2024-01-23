import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import csv
from scipy.fft import fft,fftfreq,ifft
from scipy.interpolate import interp1d
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


####################################################### PCA ####################################################################

# Hierarchical coordinate system based on data to respresent statistical variation in the data

# X= [[ ------ experiment 1 -------], [------ experiment 2 -------], .....]

# Step 1:   Compute mean row X'= 1/n (sum of row)
#           Average matrix= 1 [---- X'----]
#                           1
#                           1
#
# Step 2:   Subtract mean B= X- X'
#
# Step 3:   Covariance matrix of rows of B:
#           C= B^T B
#
# Step 4:   Compute eigenvalues and eigenvectors
#           CV= VD ------> V: Eigenvectors
#           T= BV -----> T: principal components, V: Loadings 


# PCA is used to identify dominant directions of variance in that data

# The order in which gesture features are added to the dataset
names=['eye', 'gazeleft', 'Gazeright','Open', 'Pulll', 'Pullr', 'Relax', 'Updown']


# Read the dataset for which PCA needs to be obtained
Y= pd.read_csv("Saadia/Ensemble/dataset_upper_face.csv", sep=',', header=None)

# Get the pca components 
i= len(Y.values)-1     
pca_model= utils.get_pcacomponents(Y.values, i)
pca_components= pca_model.components_
values= Y.values[1:]
data= []
colnames= []
count= 0
for x in values:
    data.append(utils.get_features(x, pca_components))
    if(count<=i):
        colnames.append(count)
    count= count + 1

with open("Saadia/Ensemble/features_upper_face"+str(i)+".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write array data
        writer.writerow(colnames)
        writer.writerows(data)


############################################################ FFT ################################################################

# with open("Saadia/EMG/Data/dataset/dataset_not_padded.csv") as fp:
#     line= fp.readlines()

# # Get the longest line
# max=0
# print(type(max))
# for l in line:
#     if(len(l) >= max):
#         max= len(l)

# data= []
# colnames= []
# count= 0

# for j in range(max):
#      colnames.append(count)
#      count= count+1

# for i in line:
#     # Get the numbers out of the string
#     features= i.split(',')
#     # Create an array out of the numbers
#     feature_array= np.array(list(map(float, features)))
#     feature_array_time= np.linspace(0, max-1, len(feature_array))

#     # Interpolate the signal to match the maximum length signal
#     interp_func= interp1d(feature_array_time, feature_array, kind='linear')
#     interpolated_time= np.linspace(0, feature_array_time[-1], max)
#     interpolated_signal= interp_func(interpolated_time)

#     # Get the fft of the data
#     fft_vals= fft(interpolated_signal)
#     dumping_fft_vals= 10*np.log10(abs(fft_vals))
#     data.append(dumping_fft_vals)


# with open("Saadia/EMG/Processed Data/features_fft.csv", 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         # Write array data
#         writer.writerow(colnames)
#         writer.writerows(data)


# ############################################################ LDA ################################################################

# dataset= pd.read_csv("Saadia/EMG/Processed Data/features.csv", sep=',', header=None)

# # divide the dataset into class and target variable
# X = dataset.iloc[:, 0:26].values
# y = dataset.iloc[:, 26].values

# # Preprocess the dataset and divide into train and test
# sc = StandardScaler()
# X = sc.fit_transform(X)
# le = LabelEncoder()
# y = le.fit_transform(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # apply Linear Discriminant Analysis
# lda = LinearDiscriminantAnalysis(n_components=4)
# X_train = lda.fit_transform(X_train, y_train)
# X_test = lda.transform(X_test)

# # plot the scatterplot
# plt.scatter(
# 	X_train[:,0],X_train[:,1],c=y_train,cmap='rainbow',
# alpha=0.7,edgecolors='b'
# )
# plt.show()

# # # classify using random forest classifier
# # classifier = RandomForestClassifier(max_depth=2, random_state=0)
# # classifier.fit(X_train, y_train)
# # y_pred = classifier.predict(X_test)

# # # print the accuracy and confusion matrix
# # print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
# # conf_m = confusion_matrix(y_test, y_pred)
# # print(conf_m)

# from sklearn.svm import SVC
# # Y_train_= np.argmax(Y_train_, axis=1)
# # Y_test= np.argmax(Y_test, axis= 1)
# clf = SVC( kernel='linear' , gamma ='auto',C=2)

# clf.fit(X_train, y_train)

# Y_pred_SVM = clf.predict((X_test))

# print("Test Accuracy for 1st Person :  "+str(round(accuracy_score(y_test,Y_pred_SVM)*100 , 2))+' %')

# conf_m = confusion_matrix(y_test, Y_pred_SVM)
# print(conf_m)