import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from numpy.linalg import norm
from sklearn import svm
from mlxtend.plotting import plot_decision_regions

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

def displayData(X,y, grid=False, clf=None):
    pos = y == 1
    neg = y == 0
    plt.plot(X[pos, 0], X[pos, 1], 'X', mew=1, ms=10, mec='k')
    plt.plot(X[neg, 0], X[neg, 1], 'o', mew=1, mfc='y', ms=10, mec='k')
    plt.grid(grid)

    if clf:
        if clf.kernel == 'linear':
            w = clf.coef_[0]
            a = -w[0] / w[1]
            xx = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 10)
            yy = a * xx - clf.intercept_[0] / w[1]
            plt.plot(xx,yy,'k-')
        elif clf.kernel == 'rbf':
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            plot_decision_regions(X, y, clf=clf, legend=2)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            # h = 0.2 # Mesh step size
            # cm = plt.cm.RdBu
            # x_min, x_max = X[:, 0].min(), X[:, 0].max()
            # y_min, y_max = X[:, 1].min(), X[:, 1].max()
            # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            #              np.arange(y_min, y_max, h))
            # if hasattr(clf, "decision_function"):
            #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            # else:
            #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # Z = Z.reshape(xx.shape)
            # plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        else:
            raise TypeError('Must be rbf or linear (for now)')

    

def svmTrain(X, y, C, kernel='linear'):
    y = y.astype(int)

    # They want me to use a simplified model they wrote themselves
    # Im just gonna use sklearn SVM packages...

    clf = svm.SVC(kernel=kernel, C = C)
    clf.fit(X,y)
    return clf


