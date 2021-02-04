# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:39:08 2021

@author: Jerry
"""
from sklearn.datasets import make_blobs,make_circles,make_moons
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import cycle, islice
import time
from k_means_clustering import KMeans
from fuzzy_c_means_clustering import FCM
#%%
# Generate sample data
datasets = []

n_samples = 1500
random_state = 170
X, _ = make_blobs(n_samples=n_samples, random_state=random_state)
datasets.append(X)

#noise moon
X,_ = make_moons(n_samples=n_samples, noise=.05)
datasets.append(X)

# Anisotropicly distributed data
X, _ = make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)
datasets.append(X)

#circle
X, _  = make_circles(n_samples=n_samples, factor=.5,
                     noise=.05)
datasets.append(X)
#%%
def min_max_normalize(array):
    return (array-np.min(array,axis=0))/(np.max(array,axis=0)-np.min(array,axis=0))

plt.figure(figsize=(6 * 1.3 + 2, 10))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
plot_num = 1


for i, dataset in enumerate(datasets):
    
    normalized_X = min_max_normalize(dataset)
    
    # ============
    # Create cluster objects
    # ============
    
    kmean = KMeans(n_clusters=3)
    fcm = FCM(n_clusters=3)
  
    
    clustering_algorithms = (
        ("KMean",kmean),
        ("fuzzy c-means",fcm)
    )
    
    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        algorithm.fit(normalized_X)
        t1 = time.time()
        
        y_pred = algorithm.predict()
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
       
        if i == 0:
            plt.title(name, size=20)
        
       
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(normalized_X[:, 0], normalized_X[:, 1], s=10, color=colors[y_pred])
        plt.scatter(algorithm.cluster_centers[:,0],algorithm.cluster_centers[:,1],c="red",marker="*",s=80)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
    
plt.show()