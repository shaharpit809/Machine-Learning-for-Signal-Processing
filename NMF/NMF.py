# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:00:58 2018

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


def update_W(S,W,H,ones_matrix):    
    num_W = np.dot(np.divide(S , np.dot(W , H)) , H.T)
    den_W = np.dot(ones_matrix , H.T)
    W = np.multiply(W , np.divide(num_W , den_W))
    return W

def NMF(S,W,flag):    
    H = np.random.uniform(0,1,S.shape[1])    
    for i in range(W.shape[1] - 1):
       H = np.vstack((H,np.random.uniform(0,1,S.shape[1])))
    
    ones_matrix = np.ones(shape = (S.shape[0],S.shape[1]))
    
    count = 0
    backup_error = 0
    app = 0
    while True:
        W[W == 0] = 0 + 1e-20
        H[H == 0] = 0 + 1e-20
        
        if flag:
            W = update_W(S,W,H,ones_matrix)
        
        num_H = np.dot(W.T , np.divide(S , np.dot(W , H)))
        den_H = np.dot(W.T , ones_matrix)
        H = np.multiply(H , np.divide(num_H , den_H))
    
        count+=1
        
        prod = np.dot(W,H)
        
        error = np.sum((S * np.log(S / prod)) - S + prod)        
        app = np.append(app , error)
        if backup_error != 0:
            if (backup_error - error) < 0.01:
                print(error - backup_error)
                break
        backup_error = error
    plt.plot(app[1:])
    print(count)
        
    return W,H

W = np.random.uniform(0,1,30)    
for i in range(512):
    W = np.vstack((W,np.random.uniform(0,1,30)))

conj_trs,magnitude_matrix_trs = STFT(trs,N)
W_S,H_S = NMF(magnitude_matrix_trs,W,True)

conj_trn,magnitude_matrix_trn = STFT(trn,N)
W_N,H_N = NMF(magnitude_matrix_trn,W,True)

WS_WN = np.hstack((W_S,W_N))

conj_x_nmf,magnitude_matrix_x_nmf = STFT(x_nmf,N)
W_SN, H_SN = NMF(magnitude_matrix_x_nmf,WS_WN,False)

mask = np.divide(np.dot(W_S ,H_SN[0:30]) ,np.dot(W_SN ,H_SN))

s_dash = np.multiply(mask , conj_x_nmf)

conj = np.conjugate(s_dash)

#s_dash = np.vstack((s_dash,conj))
for i in range(511,0,-1):
    s_dash = np.vstack((s_dash,conj[i]))

def IDFT(x,N):
    
    f=[]
    for i in range(0,N):
        f.append(i)
    f=np.array(f)
    f=f.reshape(f.shape[0],1)
        
    #creating n matrix of size N from 0-N
    n=f.T

    recover_dft = (1/N)* (np.exp(2j * np.pi * f * (n/N)))
    
    s_dash = np.dot(recover_dft,x)
    s_dash_T = s_dash.T
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

output = IDFT(s_dash,N)
librosa.output.write_wav('recovered_file.wav', (output.real).T, sr)
