# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:07:58 2018

@author: arpit
"""

import os

os.chdir('D:\IUB\Machine Learning for Signal Processing\Assignment\Assignment-3\data')

import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy import signal

trs, sr = librosa.load('trs.wav', sr=None)
trn, sr = librosa.load('trn.wav', sr=None)
x_nmf, sr = librosa.load('x_nmf.wav', sr=None)

N=1024

def STFT(x,N):
    
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
    
    #Only taking 513 rows from 1024
    X = matrix_Y[0:513,:]
    S = np.abs(X)
    return X,S

conj_trs,magnitude_matrix_trs = STFT(trs,N)

conj_trn,magnitude_matrix_trn = STFT(trn,N)

conj_x_nmf,magnitude_matrix_x_nmf = STFT(x_nmf,N)

G = conj_trs + conj_trn

G_abs = np.abs(G)

B = np.zeros(shape = (G.shape[0],G.shape[1]))

for i in range(0,G.shape[0]):
    for j in range(0,G.shape[1]):
        if(magnitude_matrix_trs[i][j] > magnitude_matrix_trn[i][j]):
            B[i][j] = 1

def KNN(G,Y,B,k):
    Dist = np.zeros(shape = (Y.shape[1] , G.shape[1]))
    for i in range(Y.shape[1]):#131
        for j in range(G.shape[1]):#987
            diff = np.sum(np.power(Y[:,i] - G[:,j] , 2))
            distance = np.sqrt(diff)
            Dist[i][j] = distance
    
    sorted_distance = Dist.argsort()
    
    k_index = sorted_distance[:,0:k]
    
    reconstruct_matrix = np.zeros((Y.shape[1],Y.shape[0]))
    for i in range(0,k_index.shape[0]):
        B_all = np.zeros((513,1))
        for j in range(0,k_index.shape[1]):
            col_index = int(k_index[i][j])
            median_value = B[:,col_index]
            median_value =  median_value.reshape((median_value.shape[0],1))
            B_all = np.hstack((B_all,median_value))
        B_all = B_all[:,1:]
        median_k = np.median(B_all, axis = 1)
        reconstruct_matrix[i,:] = median_k    
    
    r_T = reconstruct_matrix.T
            
    S_test = np.multiply(r_T,conj_x_nmf)     
    
    conj_Stest = np.conjugate(S_test)

    for i in range(511,0,-1):
        S_test = np.vstack((S_test,conj_Stest[i]))
     
    return S_test

def IDFT(x,N):    
     
    f=[]
    for i in range(0,N):
        f.append(i)
    f=np.array(f)
    f=f.reshape(f.shape[0],1)
    n=f.T
     
    recover_dft = (1/N)* (np.exp(2j * np.pi * f * (n/N)))
     
    s_dash1test = np.dot(recover_dft,x)
     
    s_dash_T = s_dash1test.T
    i=1
    output = s_dash_T[0,0:1024]
    output = output.reshape(np.shape(output)[0],1).T
    for i in range(np.shape(s_dash_T)[0]):
        first_half = s_dash_T[i-1,512:1024]
        first_half = first_half.reshape(np.shape(first_half)[0],1)
        second_half = s_dash_T[i,0:512]
        second_half = second_half.reshape(np.shape(second_half)[0],1)
        addition = first_half.T + second_half.T
        output = np.hstack((output,addition))
    
    return output

S_test = KNN(G_abs,magnitude_matrix_x_nmf,B,15)
output = IDFT(S_test,N)     
librosa.output.write_wav('output.wav', (output.real).T, sr)