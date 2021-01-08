import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def display(X, indices=None, figsize=(10, 10)):
    if X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    elif X.ndim ==2:
        m, n = X.shape
    else:
        raise IndexError('Should be 1D or 2D array of pixels')

    side_width = int(np.sqrt(n)) # add check that this is integer

    # Compute number of items to display
    show_indices = indices or range(m)
    display_rows = int(np.floor(np.sqrt(len(show_indices))))
    display_cols = int(np.ceil(len(show_indices) / display_rows))
    
    
    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)
    
    ax_array = [ax_array] if m == 1 else ax_array.ravel()
    # print(ax_array)

    for i, ax in enumerate(ax_array):
        ax.imshow(X[show_indices[i]].reshape(side_width, side_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

def display_single(X, index=0):
    """
    Display one image from pixels
    """
    if X.ndim == 1:
        n = X.size
        m = 1
    elif X[index].ndim == 1:
        X = X[index]
        n = X.size
        m = 1
    else:
        raise IndexError('Should be 1D array of pixels')

    side_width = int(np.sqrt(n)) # add check that this is integer
    plt.figure()
    plt.imshow(X.reshape(side_width, side_width, order='F'),
            cmap='Greys', extent=[0,1,0,1])
    plt.axis('off')
    
def lrCostFunction(theta, X, y, lamb):
    if y.dtype == bool:
        y = y.astype(int)
    m = len(y)
    thetaX = np.dot(X, theta)
    sig = sigmoid(thetaX)
    one_min_y = np.subtract(1,y)
    one_min_sig = np.subtract(1,sig)
    J = -(1/m)*(np.dot(y,np.log(sig))+(np.dot(one_min_y,np.log(one_min_sig))))
    reg = (lamb/(2*m))*np.dot(theta[1:],theta[1:])
    J = J + reg

    return J

def lrGradFunction(theta, X, y, lamb):
    if y.dtype == bool:
        y = y.astype(int)
    m = len(y)
    thetaX = np.dot(X, theta)
    sig = sigmoid(thetaX)
    sig_min_y = np.subtract(sig, y)
    grad = np.dot((1/m), np.dot(sig_min_y, X))
    grad_reg = np.dot(lamb/m, theta[1:])
    grad = np.add(grad, np.append([0], grad_reg))
    return grad

def NNpredict(theta1, theta2, X):
    # X can be 1D or 2D array
    if X.ndim == 1:
        one_d = True
        X = np.expand_dims(X, axis=0)
    elif X.ndim > 2 or X.ndim==0:
        raise IndexError('Must be 1D or 2D')

    # I assume 1s have already been added initially
    z1 = np.matmul(X, theta1.T)
    layer2 = sigmoid(z1)
    add_ones = np.ones((X.shape[0],1))
    layer2 = np.hstack((add_ones, layer2))
    z2 = np.matmul(layer2, theta2.T)
    layer3 = sigmoid(z2)
    
    prediction = [np.where(arr == np.amax(arr))[0][0] for arr in layer3]
    return prediction