# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:46:34 2018

@author: arpit
"""

import os

os.chdir('D:\IUB\Machine Learning for Signal Processing\Assignment\Assignment-3\data')


import librosa
import numpy as np
import matplotlib.pyplot as plt

input = np.zeros((76800,20))
for i in range(0,20):
    s,sr = librosa.load('x_ica_' + str(i+1) +'.wav', sr=None)
    input[:,i] = s
    
x_matrix = input.T


#THis function is used to calculate the singular values
def singular_value_func(eigen_vector,residual_mat):
    sd = np.sqrt(np.sum((np.dot(eigen_vector.T,residual_mat)) ** 2))
    
    return sd
 
#This method is used to calculate the eigenvectors
def power_iteration(matrix,eigenvector,iterations):
    
    for i in range(iterations):
        temp_eigenvector = np.dot(matrix,eigenvector)
        
        norm_constant = np.sqrt(np.sum(temp_eigenvector ** 2)) #Calculating the normalising constant
        computed_eigenvector = temp_eigenvector/norm_constant #Computing the eigen vector
        eigenvector = computed_eigenvector
    return computed_eigenvector

#Used to calculate the right basis vectors
def u_matrix_func(residual_mat,eigen_vector,singular_value):
    ud = np.dot(residual_mat.T,eigen_vector)/singular_value 
    shaped_ud = ud.reshape(ud.shape[0],1)
    
    return shaped_ud

#Used to calculate V*S*U matrix ie. left singular matrix * singular values * right singular matrix
def intermediate_mat_func(eigenvector,singular_value,u_mat):
    itermediate_mat = np.dot(eigenvector,singular_value)
    computed_mat = np.dot(itermediate_mat,u_mat.T)

    return computed_mat

#USed to calculate the residual matrix
def residual_mat_func(matrix,computed_mat):
    residual_matrix = matrix - computed_mat
    
    return residual_matrix

def covariance(B):
	#print(B.shape[1])
	#print(np.shape(B.mean(axis=0)))
	A_mean = B - B.mean(axis=1).reshape(B.shape[0],1)
	N=float(B.shape[1] -1)
	return np.dot(A_mean,A_mean.T) / N

#This method returns all the 8 eigenvectors
def eigenvectors_func(matrix):
#Calculating all the nececssary values for first eigenvector
    S=[]
    #Calculating the symmetric covariance matrix
#    b_mat = matrix - matrix.mean(axis = 1).reshape(matrix.shape[0],1)
#    transpose_mat = b_mat.T
#    covariance_mat = np.dot(b_mat,transpose_mat)/(b_mat.shape[1]-1) 
    covariance_mat = covariance(x_matrix)
    #Taking matrix of ones
    nrows = np.size(matrix,0)
    ones_mat = np.ones((nrows,1)) 
    
    #Calculating the first eigen vector
    first_eigenvector = power_iteration(covariance_mat,ones_mat,100)
    shaped_first_eigenvector = first_eigenvector.reshape(first_eigenvector.shape[0],1)
    
    #Calculating singular value
    singular_value = singular_value_func(shaped_first_eigenvector,covariance_mat) ##
    S.append(singular_value)
    #Calculating the right basis vectors
    u_matrix = u_matrix_func(covariance_mat,first_eigenvector,singular_value)
    
    #Calculating V*S*U  
    computed_mat = intermediate_mat_func(shaped_first_eigenvector,singular_value,u_matrix)
    
    #Calculating residual matrix
    residual_matrix = residual_mat_func(covariance_mat,computed_mat)
    
    #Stacking the first eigenvector
    stack = np.hstack((shaped_first_eigenvector))
    stack = stack.reshape(stack.shape[0],1)
    
##Calculating all the nececssary values for remaining eigenvectors
    for i in range(1,20):
        if i == 1:
            eigen_vector = power_iteration(residual_matrix,shaped_first_eigenvector,100)
        else:
            eigen_vector = power_iteration(residual_matrix,eigen_vector,100)
        stack = np.hstack((stack,eigen_vector))
        singular_value = singular_value_func(eigen_vector,residual_matrix)
        u_matrix = u_matrix_func(residual_matrix,eigen_vector,singular_value)
        intermediate_mat = intermediate_mat_func(eigen_vector,singular_value,u_matrix)
        residual_matrix = residual_mat_func(residual_matrix,intermediate_mat)
        S.append(singular_value)
    #Returning all the eigenvectors clubbed together
    return stack,S

eigenvectors,S_array = eigenvectors_func(x_matrix)

#op = np.dot(eigenvectors.T,x_matrix)

plt.matshow(eigenvectors.T,aspect='auto')
plt.title('Eigenvectors for input matrix' )
plt.show()
S_array = np.asarray(S_array)
sq_S = np.sqrt(S_array)
Lambda = np.diag(1/np.sqrt(S_array))


Intermediate_W = np.dot(Lambda,eigenvectors.T)

Whiten_Data = np.dot(Intermediate_W,x_matrix)

original = covariance(Whiten_Data)

one_inter = np.dot(Intermediate_W,covariance(x_matrix))
two_inter = np.dot(np.dot(one_inter,eigenvectors),Lambda)

i_matrix = np.identity(4)
i_matrix = (x_matrix.shape[1]) * i_matrix
err = []
W_matrix = np.identity(4)
backup=np.zeros(shape=(4,4))
count = 0
Y_matrix = np.dot(W_matrix,Whiten_Data[0:4:,])
while True:    
    delta_W_matrix = np.dot(i_matrix - np.dot(np.tanh(Y_matrix),(Y_matrix ** 3).T),W_matrix)
    W_matrix = W_matrix + (0.000001 * delta_W_matrix)
    Y_matrix = np.dot(W_matrix,Whiten_Data[0:4:,])
    count+=1
    err.append(np.max(np.abs(backup - delta_W_matrix)))
#    if np.all(W_matrix==backup):
#        break
#    backup = deepcopy(W_matrix)
    if(count == 10000):
        break
    backup = delta_W_matrix

final_vav = Y_matrix[0]
for i in range(1,4):
    final_vav = np.hstack((final_vav,Y_matrix[i]))

plt.plot(err)  
plt.show()  
#librosa.output.write_wav('final.wav', final_vav, sr)
librosa.output.write_wav('output1.wav', Y_matrix[0], sr)
librosa.output.write_wav('output2.wav', Y_matrix[1], sr)
librosa.output.write_wav('output3.wav', Y_matrix[2], sr)
librosa.output.write_wav('output4.wav', Y_matrix[3], sr)

