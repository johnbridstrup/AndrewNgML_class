import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def createLambda(f, *args):
    return lambda p: f(p, *args)

def linearRegCost(theta, X, y, lamb=0):
    m = y.size+1
    grad = np.zeros(theta.shape)

    h = X.dot(theta)
    J = (1/(2*m)) * np.sum(np.square(h-y)) + (lamb/m)*np.sum(np.square(theta[1:]))

    grad = (1/m) * (h-y).dot(X)
    grad[1:] = grad[1:] + (lamb/m) * theta[1:]

    return J, grad

def trainLinReg(X, y, lamb=0, opts = {'maxiter': 200}):
    init_theta = np.zeros(X[0].size)
    cfunc1 = lambda p: linearRegCost(p, X, y, lamb)
    res = opt.minimize(cfunc1,
                        init_theta,
                        jac=True,
                        method='TNC',
                        options=opts)
    return res
                        
def learningCurve(X, y, Xval, yval, lamb=0):
    m = y.size
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1, m+1):
        res = trainLinReg(X[:i], y[:i], lamb)
        error_train[i-1], _ = linearRegCost(res['x'], X[:i], y[:i])
        error_val[i-1], _ = linearRegCost(res['x'], Xval, yval)
    return error_train, error_val

def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0], p))
    for i in range(p):
        X_poly[:,i] = X[:,0] ** (i+1)
    return X_poly

def featureNormalize(X):
    mean = X.mean(axis=0)
    X_norm = X - mean.T

    std = X.std(axis=0)
    X_norm = X_norm / std
    return X_norm, mean, std

def plotFit(min_x, max_x, mu, sigma, theta, p):
    # make range a little bigger
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)
    X_poly = polyFeatures(x, p)
    X_poly -= mu.T
    X_poly /= sigma

    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)
    plt.plot(x, np.dot(X_poly, theta), '--', lw=2)
