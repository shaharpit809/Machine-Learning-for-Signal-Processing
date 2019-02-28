# -*- coding: utf-8 -*-
"""
@author: arpit
"""

import os

os.chdir('D:\IUB\Machine Learning for Signal Processing\Assignment\Assignment-2\data\data')

import numpy as np
import scipy.io as sio

#Reading june.mat and decemeber.mat
june = sio.loadmat('june.mat')
december = sio.loadmat('december.mat')

june_mat = june['june']
december_mat = december['december']

#Calculating the disparity matrix
disparity_x = december_mat[:,0] - june_mat[:,0]
disparity_x = disparity_x.reshape(disparity_x.shape[0],1)

#Initialising the mean with values that we got from K-means
mean_mat = np.array([24,41])
mean_mat = mean_mat.reshape(mean_mat.shape[0],1)

#Initialising standard deviations randomly
standard_deviation = np.array([0.4,0.5])
standard_deviation = standard_deviation.reshape(standard_deviation.shape[0],1)

#Initialising the priors with 0.5 value for both of them
P = np.array([0.5,0.5],dtype=float)
P = P.reshape(np.shape(P)[0],1)

#Initialising U matrix
matrix_U = np.zeros(shape = (disparity_x.shape[0],2))

#Initialising N matrix which will consist of norm values
matrix_N = np.zeros(shape = (disparity_x.shape[0],2))

#Initialisng this matrix which will compute the values of U*prior values
U_prior = np.zeros(shape = (disparity_x.shape[0],2))

#Initialising old likelihood value with a large negative value
old_loglikelihood = -99999

#count = 0
while True:
#E-Step
    for i in range(0,matrix_N.shape[0]):
            for j in range(0,matrix_N.shape[1]):
                matrix_N[i][j] = (1 / (np.sqrt(2 * np.pi * standard_deviation[j,0])) * np.exp(-np.square(disparity_x[i,0] - mean_mat[j,0]) / (2 * standard_deviation[j,0])))
                if matrix_N[i][j] == 0:
                    matrix_N[i][j] = 0.0001    
    
    #Multiplying the priors with U matrix
    for i in range(0,2):
        U_prior[:,i] = P[i][0] * matrix_N[:,i]
    
    #Calculating the sum of of the above matrix    
    deno = np.sum(U_prior,axis = 1)    
    
    #Computing the final U matrix which will be divided by the denominator
    for i in range(0,2):
        matrix_U[:,i] = U_prior[:,i] * (1/deno)
        
#M-Step            
    matrix_U_trans = matrix_U.T
    
    sum_mean = np.array([1 / np.sum(matrix_U[:,i]) for i in range(matrix_U.shape[1])])
    sum_mean = sum_mean.reshape(sum_mean.shape[0],1)
    #Calculating the means
    mean_mat = np.multiply(np.dot(matrix_U.T,disparity_x),sum_mean)
    #Calculating the standard deviations
    standard_deviation = np.array([np.dot(matrix_U_trans[i,:],np.square((disparity_x - mean_mat[i,0]))) * sum_mean[i][0] for i in range(matrix_U.shape[1])])
    
#    count +=1
#    print(count)
    #Calculating the probabilities
    P = np.array([sum(matrix_U[:,i])/np.shape(matrix_U)[0] for i in range(matrix_U.shape[1])])
    P = P.reshape(np.shape(P)[0],1)    
    
    #If thee are any value in the U matrix that is equal to 0 then we change it
    #to a small value because log of 0 is inf    
    for i in range(0,U_prior.shape[0]):
        for j in range(0,U_prior.shape[1]):
            if U_prior[i][j] == 0:
                U_prior[i][j] = 0.01
        
    #Calculating the likelihood   
    new_loglikelihood = np.sum(np.log(U_prior))
    #Calculating the difference of new likelihood with the previous one 
    if np.abs(new_loglikelihood - old_loglikelihood) < 0.01:
        break
    else:    
        old_loglikelihood = new_loglikelihood
        
print(mean_mat)
#[[20.8432586 ]
# [40.15283258]]