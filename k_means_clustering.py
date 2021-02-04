# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 19:49:23 2021

@author: Jerry
"""

from sklearn.datasets import make_blobs,make_circles,make_moons
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import cycle, islice
import time
#%%
class KMeans:
    def __init__(self,n_clusters=3,max_iters=10):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
    
    def fit(self,X):
        self.X = X
        self.n_samples = len(X)
        self.clusters = np.zeros(self.n_samples,dtype=int)
        self.cluster_centers = np.array(random.choices(X,k=self.n_clusters))
        
        for i in range(self.max_iters):
            self.assign_cluster()
            self.update_cluster_centers()
             
    def assign_cluster(self):
        for i,point in enumerate(self.X):
            self.clusters[i] = np.argmin(np.sum(np.square(point-self.cluster_centers),axis=1))
        
    def update_cluster_centers(self):
        for i in range(self.n_clusters):  
            self.cluster_centers[i] = np.mean(self.X[self.clusters==i],axis=0)
            
    def predict(self):
        return self.clusters

#%%
        