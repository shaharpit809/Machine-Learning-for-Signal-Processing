# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 18:55:55 2018

@author: arpit
"""

import os

os.chdir('D:\IUB\Machine Learning for Signal Processing\Assignment\Assignment-3\data')

import numpy as np
import scipy.io as sio

eeg = sio.loadmat('eeg.mat')

x_train = eeg['x_train']
x_test = eeg['x_te']

y_train = eeg['y_train']
y_test = eeg['y_te']

N = 64
l = 110
k = 21
def STFT(N,x):
    
    blackman_window = np.blackman(N)
    blackman_window = blackman_window.reshape(np.shape(blackman_window)[0],1)

    #Taking sample of N size after every N/2 intervals and then multiplying it with the Hann's window
    #After the multiplication, stacking it into the X matrix
    for i in range(0,x.shape[2]):    
        h = np.array([1000000])
        h = h.reshape((h.shape[0],1))
        for j in range(0,x.shape[1]):   
            matrix_X = np.zeros((N,1))
            something1 = x[:,j]
            something2 = something1[:,i]  
            for k in range(0,x.shape[0],48):
                if np.shape(something2[k:N+k])[0] == 64:
                    sample_window = something2[k:N+k]
                    sample_window = sample_window.reshape(np.shape(sample_window)[0],1)
                    intermediate_matrix = np.multiply(sample_window,blackman_window)
                    intermediate_matrix = intermediate_matrix.reshape(np.shape(intermediate_matrix)[0],1)
                    matrix_X = np.hstack((matrix_X,intermediate_matrix))
            final_matrix_X = matrix_X[:,1:]
            part_of_matrix_X = final_matrix_X[3:8,:]
            
            for l in range(0,5):
                inter = part_of_matrix_X[l]
                inter = inter.reshape((inter.shape[0],1))
                h = np.vstack((h,inter))

            if(h[0] == 1000000):
                h = h[1:]
        
        if i == 0:
            final = np.zeros((len(h),1))
            final = np.hstack((final,h))
        else:
            final = np.hstack((final,h))
    
    final = final[:,1:]
    return final

def calculating_A(l,M):
    
    np.random.seed(2)
    A = np.random.uniform(-1,1,l)
    for i in range(len(M) - 1):
        A = np.vstack((A,np.random.uniform(-1,1,l)))
    
    sum_A = [sum(A[i]) for i in range(len(M))]
    sum_A = np.array(sum_A)
    inv_sum_A = 1/sum_A
    
    B = A * inv_sum_A[:, np.newaxis]    
    B_T = B.T

    return B_T

def calculating_Y(A,Z):
    Y = np.dot(A,Z)
    
    Y_sign = np.sign(Y)
    
    return Y,Y_sign

def hamming(s1,s2):
    dist = np.count_nonzero(s1 != s2)
    return dist

def distance(Y,Y_test):
    
    distance = np.zeros((28,112))
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            distance[i][j] = hamming(Y_test[:,i],Y[:,j])
    
    sorted_distance = distance.argsort()
    final_index = np.zeros((Y_test.shape[1],Y.shape[1]))
    for i in range(Y_test.shape[1]):
        for j in range(Y.shape[1]):
            index = sorted_distance[i][j]
            final_index[i][j] = y_train[index,0]
            
    return final_index

Z = STFT(N,x_train)

Z_test = STFT(N,x_test)
list_i = []
list_j = []
list_acc = []
for i in range(10,110,5):
    for j in range(3,21,2):
        l = i
        k = j
        
        A = calculating_A(l,Z)

        Y,Y_sign = calculating_Y(A,Z)
        
        Y_test,Y_test_sign = calculating_Y(A,Z_test)
        
        index_mat = distance(Y_sign,Y_test_sign)
        k_index_mat = index_mat[:,0:k]        
        
        final_y_test = np.zeros((y_test.shape[0],1))
        for p in range(0,28):
            final_y_test[p] = np.median(k_index_mat[p,:])

        count = 0    
        for p in range(0,28):
            if final_y_test[p] == y_test[p]:
                count+=1

        accuracy = count/28

        list_i.append(i)

        list_j.append(j)

        list_acc.append(accuracy)
#        print('Accuracy for K=' + str(k) +' and L=' +  str(l) + ' is ' +str(count/28))

list_i = np.array(list_i)
list_i = list_i.reshape((list_i.shape[0],1))

list_j = np.array(list_j)
list_j = list_j.reshape((list_j.shape[0],1))

list_acc = np.array(list_acc)
list_acc = list_acc.reshape((list_acc.shape[0],1))

intermediate_stack = np.hstack((list_i,list_j))
final_stack = np.hstack((intermediate_stack,list_acc))

sorted_stack = final_stack[final_stack[:,2].argsort()]

reverse_stack = sorted_stack[::-1]

partial_reverse_stack = reverse_stack[0:20]

for i in range(len(partial_reverse_stack)):
    print('Accuracy for L=' + str(reverse_stack[i,0]) + ' and K=' + str(reverse_stack[i,1]) + ' is ' + str(reverse_stack[i,2]))