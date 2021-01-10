import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as opt

def formatData(data):
    """Format data with boolean values for categorical data"""
    # Binary data first
    binData = data.replace({'M':1, 'B':0})
    # Category data
    y = binData.diagnosis
    drop_unnamed = binData.drop('Unnamed: 32', axis=1)
    outData = drop_unnamed.drop('diagnosis', axis=1).drop('id',axis=1)
    # Normalize float data (min/max)
    # cols_to_norm = ['age','bmi']
    # outData[cols_to_norm] = MinMaxScaler().fit_transform(outData[cols_to_norm])
    # add offset feature (identical to 1)
    outData.insert(0, 'offset', 1)
    return [y,outData]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def guesses(theta, x):
    h = []
    try:
        features = x.keys()
        for _, row in x.iterrows():
            h.append(sigmoid(sum([theta[i]*row[feature] for i,feature in enumerate(features)])))
    except:
        for row in x:
            h.append(sigmoid(sum([theta[i]*xx for i,xx in enumerate(row)])))
    return h

def cost(y, h):
    if y==0:
        output = -np.log(1-h)
    elif y==1:
        output = -np.log(h)
    return output

def costFunction(theta, x, y):
    h = guesses(theta, x)
    features = x.keys()
    # print(h)
    m = len(y)
    J = sum([cost(yy,hh) for yy, hh in zip(y,h)])/m

    dTheta = []
    for feature in features:
        dth = sum([(guess - act)*row[feature] for guess, act, (index, row) in zip(h, y, x.iterrows())])
        dTheta.append(dth)

    return J, dTheta

# Load data
path = 'data/breast_cancer.csv'
data = pd.read_csv(path)
formatted_data = formatData(data)
# print(formatted_data[1].keys())
# print(formatted_data[0])

X = formatted_data[1]
y = formatted_data[0]
theta = [0 for _ in X.keys()]

print('Initial cost:')
print(costFunction(theta, X, y)[0])
print('Minimizing...')
result = opt.fmin_tnc(func=costFunction, x0=theta, args=(X, y))
print('Final cost:')
print(costFunction(result[0], X, y))


