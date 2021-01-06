import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from functions import costFunction

def inflectionPoint(theta, x):
    """This will only work for two features"""
    return -(theta[0]+theta[1]*x)/theta[2]

data_path = 'ex2data1.txt'
data = pd.read_csv(data_path, header=None, names=['test1','test2','result'])

# pos = data.loc[data['result'] == 1]
# neg = data.loc[data['result'] == 0]

y = data.result
X = data.drop('result',axis=1)
X.insert(0, 'offset', 1)
theta = [0 for _ in X.keys()]

cost, gradient = costFunction(theta, X, y)
print('Initial cost: {}'.format(cost))
print('Initial gradient: {}'.format(gradient))

result = opt.fmin_tnc(func=costFunction, x0=theta, args=(X, y))

fcost, gradient = costFunction(result[0], X, y)
print('Final cost: {}'.format(fcost))
print('Final theta: {}'.format(result[0]))

t1minmax = [30, 100]
t2DB = [inflectionPoint(result[0],30), inflectionPoint(result[0],100)]

plt.figure()
for _, row in data.iterrows():
    if row.result == 1:
        plt.scatter(row.test1, row.test2, marker='+', color='g')
    else:
        plt.scatter(row.test1, row.test2, marker='_', color = 'r')
plt.plot(t1minmax, t2DB)
plt.xlabel('test 1 score')
plt.ylabel('test 2 score')
plt.show()
