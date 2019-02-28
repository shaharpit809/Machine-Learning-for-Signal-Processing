# -*- coding: utf-8 -*-
"""
@author: arpit
"""

import os

os.chdir('D:\IUB\Machine Learning for Signal Processing\Assignment\Assignment-2\data\data')

import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal

#Loading X.wav
x, sr = librosa.load('x.wav', sr=None)
N=1024

#Declaring F matrix of NxN
matrix_F = np.zeros(shape=(N,N))
#creating f matrix of size N from 0-N
f=[]
for i in range(0,N):
    f.append(i)
f=np.array(f)
f=f.reshape(f.shape[0],1)

#creating n matrix of size N from 0-N
n=f.T

#Computing F matrix
matrix_F = np.exp(-2 * 1j * np.pi * f * (n/N))

plt.imshow(matrix_F.real)
plt.title('Real part of F matrix')
plt.show()

plt.imshow(matrix_F.imag)
plt.title('Imaginary part of F matrix')
plt.show()
#Creating Hann's window of size N
hann_window = signal.hann(N)
hann_window = hann_window.reshape(np.shape(hann_window)[0],1)

#Initialising X matrix
matrix_X = np.zeros((N,1))

#Taking sample of N size after every N/2 intervals and then multiplying it with the Hann's window
#After the multiplication, stacking it into the X matrix
for i in range(0,len(x),512):
    if np.shape(x[i:N+i])[0] == 1024:
        sample_window = x[i:N+i]
        sample_window = sample_window.reshape(np.shape(sample_window)[0],1)
        intermediate_matrix = np.multiply(sample_window,hann_window)
        intermediate_matrix = intermediate_matrix.reshape(np.shape(intermediate_matrix)[0],1)
        matrix_X = np.hstack((matrix_X,intermediate_matrix))

#Since matrix X has extra column zeros at the start, we remove that extra column
final_matrix_X = matrix_X[:,1:(np.shape(matrix_X)[1]+1)]
#Caluclating the Y matrix
matrix_Y = np.dot(matrix_F,final_matrix_X)

plt.imshow(np.abs(matrix_Y))
plt.title('Spectogram')
plt.show()

#Taking a part of the Y matrix
subset = matrix_Y[:,0:15]

#Calculating M from samples of Y
M= np.mean(np.absolute(subset),axis=1)
M= M.reshape(np.shape(M)[0],1)

#Calulating residual magnitudes
subtract = np.absolute(matrix_Y) - M
subtract = subtract.clip(0)

#Calculating the phase of Y
phase = matrix_Y/np.absolute(matrix_Y)

mul = np.multiply(phase,subtract)

#Calculting the inverse DFT
recover_dft = (1/N)* (np.exp(2j * np.pi * f * (n/N)))

#Checking if F* x F is almost equal to identity matrix or not 
I = np.dot(recover_dft,matrix_F)

#Calculting X dash by multiplying F* with spectogram
X_dash = np.dot(recover_dft, mul)

X_dash_T = X_dash.T

#Reversing the procedure applied above. Again taking the samples of size N with an interval of N/2
#Then applying overlap and add to it
#Merging it into a single array and creating the final output without noise
i=1
output = X_dash_T[0,0:1024]
output = output.reshape(np.shape(output)[0],1).T
for i in range(np.shape(X_dash_T)[0]):
    first_half = X_dash_T[i-1,512:1024]
    first_half = first_half.reshape(np.shape(first_half)[0],1)
    second_half = X_dash_T[i,0:512]
    second_half = second_half.reshape(np.shape(second_half)[0],1)
    addition = first_half.T + second_half.T
    output = np.hstack((output,addition))

#Taking only the real part of the output file
#Writing the output wav file.    
librosa.output.write_wav('output.wav', (output.real).T, sr)