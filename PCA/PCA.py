# -*- coding: utf-8 -*-
"""
@author: arpit
"""

import os

os.chdir('D:\IUB\Machine Learning for Signal Processing\Assignment\Assignment-2\data\data')


import librosa
import numpy as np
import random
import matplotlib.pyplot as plt

#Loading the s.wav file
s,sr = librosa.load('s.wav', sr=None)

#This function takes the number of sample as input and creates a Nx8 matrix
#After randomly selecting 8 consecutive samples, N times, we return transpose(8*N) of the matrix created
def mainfunc(N):
    print(N)
    matrix = np.zeros(shape=(N,8))
        
    for i in range(0,N):
        random_num = random.choice(s)
        for j in range(0, len(s)):
            if s[j] == random_num:
                index = j
                break
        for k in range(0,8):
            matrix[i][k] = s[index + k]
            
    return matrix.T

#THis function is used to calculate the singular values
def singular_value_func(eigen_vector,residual_mat):
    sd = np.sqrt(np.sum((np.dot(eigen_vector.T,residual_mat)) ** 2)) 
    shaped_sd = sd.reshape(1,1)
    
    return shaped_sd
 
#This method is used to calculate the eigenvectors
def power_iteration(matrix,eigenvector,iterations):
    
    for i in range(iterations):
        temp_eigenvector = np.dot(matrix,eigenvector)
        
        norm_constant = np.sqrt(np.sum(temp_eigenvector ** 2)) #Calculating the normalising constant
        computed_eigenvector = temp_eigenvector/norm_constant #Computing the eigen vector
        
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

#This method returns all the 8 eigenvectors
def eigenvectors_func(matrix):
#Calculating all the nececssary values for first eigenvector
    
    #Calculating the symmetric covariance matrix
    b_mat = matrix - matrix.mean(axis = 1).reshape(matrix.shape[0],1)
    transpose_mat = b_mat.T
    covariance_mat = np.dot(b_mat,transpose_mat)/(b_mat.shape[1]-1) 
    
    #Taking matrix of ones
    nrows = np.size(matrix,0)
    ones_mat = np.ones((nrows,1)) 
    
    #Calculating the first eigen vector
    first_eigenvector = power_iteration(covariance_mat,ones_mat,100)
    shaped_first_eigenvector = first_eigenvector.reshape(first_eigenvector.shape[0],1)
    
    #Calculating singular value
    singular_value = singular_value_func(shaped_first_eigenvector,covariance_mat) ##
    
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
    for i in range(1,8):
        if i == 1:
            eigen_vector = power_iteration(residual_matrix,shaped_first_eigenvector,100)
        else:
            eigen_vector = power_iteration(residual_matrix,eigen_vector,100)
        stack = np.hstack((stack,eigen_vector))
        singular_value = singular_value_func(eigen_vector,residual_matrix)
        u_matrix = u_matrix_func(residual_matrix,eigen_vector,singular_value)
        intermediate_mat = intermediate_mat_func(eigen_vector,singular_value,u_matrix)
        residual_matrix = residual_mat_func(residual_matrix,intermediate_mat)
    
    #Returning all the eigenvectors clubbed together
    return stack

#Selecting 10 samples and then plotting    
data_matrix = mainfunc(10)
eigenvectors = eigenvectors_func(data_matrix)

plt.matshow(eigenvectors.T,aspect='auto')
plt.title('Eigenvectors for 10 samples' )
plt.show()

#Selecting 100 samples and then plotting
data_matrix = mainfunc(100)
eigenvectors = eigenvectors_func(data_matrix)

plt.matshow(eigenvectors.T,aspect='auto')
plt.title('Eigenvectors for 100 samples' )
plt.show()

#Selecting 1000 samples and then plotting
data_matrix = mainfunc(1000)
eigenvectors = eigenvectors_func(data_matrix)

plt.matshow(eigenvectors.T,aspect='auto')
plt.title('Eigenvectors for 1000 samples' )
plt.show()