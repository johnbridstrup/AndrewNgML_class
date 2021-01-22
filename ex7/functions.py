import numpy as np
from numpy.linalg import svd
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

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm, mu, sigma

def pca(X):
    """
    Principle component analysis
    implements svd from numpy.linalg
    --------------------------------------
    Parameters:
    X: array-like dataset, dimensions mxn

    Returns:
    U: array-like eigen vectors of X
    S: Diagonal array of eigen values
    """
    m,_ = X.shape
    # Computer covariance
    covar = (1/m)*(X.T.dot(X))

    U, S, _ = svd(covar)

    return U, S

def projectData(X, U, K):
    """
    Project data after principle component analysis
    ---------------------------------------
    Parameters:
    X: Array of data
    U: Svd of covariance matrix of X
    K: number of principle components to use

    Returns:
    Z: Data projected onto the K largest eigenvectors
    """
    Z = np.dot(X, U[:,:K])
    return Z

def recoverData(Z, U, K):
    X_rec = Z.dot(U[:,:K].T)
    return X_rec

def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data in a nice grid.
    Parameters
    ----------
    X : array_like
        The input data of size (m x n) where m is the number of examples and n is the number of
        features.
    example_width : int, optional
        THe width of each 2-D image in pixels. If not provided, the image is assumed to be square,
        and the width is the floor of the square root of total number of pixels.
    figsize : tuple, optional
        A 2-element tuple indicating the width and height of figure in inches.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_height, example_width, order='F'), cmap='gray')
        ax.axis('off')