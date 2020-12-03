import numpy as np
from numpy import random
import matplotlib.pyplot as plt


## Generate a straight line with random m and random b

# Function generator for a straight line
def line(m,b):
    def y(x):
        try:
            return m*x + b
        except:
            return [m*xx + b for xx in x]
    return y

# Generate slope between -5 and 5
m = 10*random.rand()-5

# Generate intercept between -5 and 5
b = 10*random.rand()-5

# Get line function and populate data
initLine = line(m,b)
x=[i for i in range(100)]
cleanData = initLine(x)

# apply noise to data
mu, sigma = 0,10
y_data = [i + random.normal(mu,sigma) for i in cleanData]

plt.figure()
plt.plot(x,cleanData)
plt.scatter(x,y_data)
plt.show()


# Apply gaussian noise to the line 
# 
# Fit line with linear regression using gradient descent