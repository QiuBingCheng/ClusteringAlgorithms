# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 17:42:33 2021

@author: Jerry
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import random
#%%
class FCM:
    """Fuzzy C-means
    
    Parameters
    ----------
    n_clusters: int, optional (default=3)
        The number of clusters to form as well as the number of
        centroids to generate.
        
    m: float, optional (default=2.0)
        This option controls the amount of fuzzy overlap between clusters, 
        with larger values indicating a greater degree of overlap
    
     Attributes
    ----------
    n_samples: int
        Number of examples in the data set
        
    n_features: int
        Number of features in samples of the data set
        
    cluster_centers: array, shape = [n_samples, n_clusters]
        Final cluster centers, returned as an array with n_clusters rows
        containing the coordinates of each cluster center
    
    """
    
    def __init__(self,n_clusters,m=2,max_iters=10):
        self.n_clusters = n_clusters 
        self.m = m
        self.max_iters = max_iters
        
    def fit(self,X):
        self.X = X
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.cluster_centers = np.array(random.choices(self.X,k=self.n_clusters))
        self.distance = np.zeros((self.n_samples,self.n_clusters))
        self.membership = np.zeros((self.n_samples,self.n_clusters))
       
        for i in range(self.max_iters):
            self.update_distance()
            self.update_membership()
            self.update_cluster_centers()         
            
    def _cal_dis(self,arr1,arr2):
        dis = np.sqrt(np.sum(np.square(arr1-arr2)))
        return dis if dis>0 else pow(10,-3)
    
    def update_distance(self):
        for i in range(self.n_samples):
            for j in range(self.n_clusters):
                self.distance[i][j] = self._cal_dis(self.X[i],self.cluster_centers[j])
    
    def _cal_membership(self,i,j):
        mem = 0
        for k in range(self.n_clusters):
            mem += np.power(self.distance[i][j]/self.distance[i][k],2/(self.m-1))
        return 1/mem     
    
    def update_membership(self):
        for i in range(self.n_samples):
            for j in range(self.n_clusters):
               self.membership[i][j] = self._cal_membership(i,j)      
    
    def update_cluster_centers(self):
        for i in range(self.n_clusters):
            mem = np.power(self.membership[:,i],self.m) #the membership vector of the cluster 
            mem2 = np.transpose([mem]*2)
            self.cluster_centers[i] = np.sum(mem2*self.X,0)/np.sum(mem)
    
    def predict(self):
        return np.argmax(self.membership,axis=1)
        
#%%
# Generate sample data
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state,cluster_std=[1.0, 2.5, 0.5])
#%%
fcm = FCM(3,max_iters=10)        
fcm.fit(X)
plt.scatter(X[:,0],X[:,1],c=fcm.predict(),s=80)