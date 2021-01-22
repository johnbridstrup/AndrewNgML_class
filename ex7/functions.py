import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import random

from IPython.display import HTML, display, clear_output

try:
    plt.rcParams["animation.html"] = "jshtml"
except ValueError:
    plt.rcParams["animation.html"] = "html5"

def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in np.arange(idx.size):
        J = np.sum(np.square(X[i]-centroids),axis=1)
        idx[i] = np.argmin(J)
    return idx

def computeCentroids(X, idx, K):
    m,n = X.shape
    centroids = np.zeros((K,n))

    for i in range(K):
        centroids[i] = np.mean(X[idx == i], axis = 0)
    
    return centroids

def runkMeans(X, K):

    kmeans = KMeans(n_clusters=K, random_state=0)
    idx = kmeans.fit_predict(X)
    return kmeans, idx

def plotkMeans(X, fit):
    k_means_centers = fit.cluster_centers_
    K = k_means_centers.shape[0]

    k_means_labels = pairwise_distances_argmin(X, k_means_centers)

    colors = []
    plt.figure()
    for _ in range(K):
        random_number = random.randint(0,16777215)
        hex_number = str(hex(random_number))
        hex_number ='#'+ hex_number[2:]
        colors.append(hex_number)
    
    for k, col in enumerate(colors):
        members = k_means_labels==k
        cluster_center = k_means_centers[k]

        plt.plot(X[members, 0], X[members, 1], 'w',
            markerfacecolor=col, marker='.', markersize=10)
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)

