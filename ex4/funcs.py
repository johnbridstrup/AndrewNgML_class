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