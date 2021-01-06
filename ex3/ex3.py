import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from functions import display, display_single

data_path = 'exfiles/ex3data1.mat'
data = loadmat(data_path)

X = data['X']
y = data['y']
y[y == 10] = 0

print(X[0])
display(X, [1000,2000,3000,4000])

display_single(X, 1000)
plt.show()

# img = Image.fromarray(data['X'][0].reshape(20,20),'L')
# img.save('test.png')
