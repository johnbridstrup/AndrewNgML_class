import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from numpy.linalg import norm

def plotData(X,y):
    # plot y=1 as X and y=0 as O on scatter plot
    return 0

def gaussianKernel(x1,x2, sigma=1.0):
    # calculate gaussian -> exp(-||(x1-x2)||^2/(2*sigma^2))
    # must be vectorized
    # x1 and x2 are the feature vector and vector of landmarks
    # sigma is the bandwidth
    return np.exp(-np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2)))

def linearKernel(x1,x2):
    return np.dot(x1,x2)

def displayData(X,y, grid=False):
    pos = y == 1
    neg = y == 0
    plt.plot(X[pos, 0], X[pos, 1], 'X', mew=1, ms=10, mec='k')
    plt.plot(X[neg, 0], X[neg, 1], 'o', mew=1, mfc='y', ms=10, mec='k')
    plt.grid(grid)

def svmTrain(X, y, C, kernelFunction, tol=1e-3, max_passes=5, args=()):
    y = y.astype(int)
    m,n = X.shape

    passes = 0
    E = np.zeros(m)
    alphas = np.zeros(m)
    b = 0

    # map y = 0 to y = -1 (Why do I have to do this?)
    y[y==0] = -1

    # Optimize linear and gaussian kernels
    if kernelFunction.__name__ == 'linearKernel':
        # Vectorized computation for the linear kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = np.dot(X, X.T)
    elif kernelFunction.__name__ == 'gaussianKernel':
        # vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X2 = np.sum(X**2, axis=1)
        K = X2 + X2[:, None] - 2 * np.dot(X, X.T)

        if len(args) > 0:
            K /= 2*args[0]**2

        K = np.exp(-K)
    else:
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                K[i, j] = kernelFunction(X[i, :], X[j, :])
                K[j, i] = K[i, j]

