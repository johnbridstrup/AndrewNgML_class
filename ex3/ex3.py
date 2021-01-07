import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
from functions import display, display_single, lrCostFunction, lrGradFunction, sigmoid
import random

data_path = 'exfiles/ex3data1.mat'
data = loadmat(data_path)

X = data['X']
y = data['y']


# Fix stupid matlab indexing 
y[y == 10] = 0

# Display 100 randomly chosen numbers
numbers_to_display = 100
display_indices = random.sample(range(5000), numbers_to_display)
display(X, display_indices)
# plt.show()

############
# lrCostFunction test found online, identical to ex3.mlx worksheet
############
# test values for the parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularization parameter
lambda_t = 3

####
# Actual test 
####

J = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = lrGradFunction(theta_t, X_t, y_t, lambda_t)

print('Cost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]')

# Now to train it
m, n = X.shape

# Add ones to the data matrix
add_ones = np.ones((5000,1))
X = np.hstack((add_ones, X))
initial_theta = np.zeros(n+1)
all_theta = np.zeros((10,n+1))

lamb=0.1
for c in range(10):
    y_binary = np.array([yy[0] for yy in y==c])
    theta = fmin_cg(lrCostFunction, initial_theta, lrGradFunction, args=(X, y_binary, lamb), full_output=True)
    all_theta[c] = theta[0]

z = np.matmul(X, all_theta.T)
sig = sigmoid(z)

predictions = [np.where(arr == np.amax(arr))[0][0] for arr in sig]

correct_count = 0
for idx, pred in enumerate(predictions):
    if pred==y[idx]:
        correct_count += 1

percent_correct = correct_count/m
print('Correct prediction percentage: {}'.format(percent_correct))

# How to find indices where y=number for later
# np.where(y==5)[0] -> [indices where y = 5]

