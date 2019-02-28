# -*- coding: utf-8 -*-
"""
@author: arpit
"""

import os

os.chdir('D:\IUB\Machine Learning for Signal Processing\Assignment\Assignment-2\data\data')

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#Readind .mat files
june = sio.loadmat('june.mat')
december = sio.loadmat('december.mat')
#Taking out hte matrix from the mat file
june_mat = june['june']
december_mat = december['december']
#Caluclating the disparity matrix
disparity_x = december_mat[:,0] - june_mat[:,0]
disparity_x = disparity_x.reshape(disparity_x.shape[0],1)
#Plotting the disparity matrix
plt.hist(disparity_x, bins =20)
plt.title('Disparity Histogram')
plt.show()

#After plotting the historgram, we can se that the 2 means are near 20 and 40
#Therefore initialising the centroids with 20 and 40
old_centroid = np.array([20,40])
old_centroid = old_centroid.reshape(old_centroid.shape[0],1)

while True:
    #Initialising these varialbles to 0
    #Sum_x will store the values from disparty matrix which are near to first mean and same for Sum_y
    sum_x = 0
    sum_y = 0
    #Rhis will keep the count of samples that belong to cluster 1 and 2 respectively
    counter_x = 0
    counter_y = 0
    
    #Initialising matrix that will hold the values of distance of each 'X' value from both the centroids
    argmin_mat = np.zeros(shape=(disparity_x.shape[0],old_centroid.shape[0]))
    
    #Initialising membership matrix which will hold the values of membership values of all the X values
    #Membership value will determine to which cluster does that point X belong to
    #If the value is set to 1 then it will belong to that cluster
    membership_matrix = np.zeros(shape=(disparity_x.shape[0],old_centroid.shape[0]))

    for i in range(0,disparity_x.shape[0]):
        #Calculating the distane of each X from both the centroids
        for j in range(0,old_centroid.shape[0]):
            argmin_mat[i][j] = np.square(disparity_x[i][0] - old_centroid[j][0])
        #Distance for which centorid is minimum, membership value will be updated to 1   
        for k in range(0,old_centroid.shape[0]):
            if argmin_mat[i][0] == min(argmin_mat[i][0],argmin_mat[i][1]):
                membership_matrix[i][0] = 1
            else:
                membership_matrix[i][1] = 1
        #Grouping all the values of all the 'X' for which the membership is 1
        if membership_matrix[i][0] == 1:
            sum_x+= disparity_x[i][0]
            counter_x+=1
        else:
            sum_y+= disparity_x[i][0]
            counter_y+=1 
            
    #Calculating new mean 
    new_centroid = np.array([sum_x/counter_x,sum_y/counter_y])
    new_centroid = new_centroid.reshape(new_centroid.shape[0],1)
    
    #if old and new mean match then the k-means algorithm stops
    if np.allclose(new_centroid,old_centroid):
        break
    else:
        old_centroid = new_centroid
        
print(new_centroid)
#24.358736
#41.226642