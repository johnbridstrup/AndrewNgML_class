import numpy as np
import pandas as pd

def sigmoid(z):
    return 1/(1+np.exp(-z))

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

def h(theta, x):
    out = []
    try:
        features = x.keys()
        for _, row in x.iterrows():
            out.append(sigmoid(sum([theta[i]*row[feature] for i,feature in enumerate(features)])))
    except:
        for row in x:
            out.append(sigmoid(sum([theta[i]*xx for i,xx in enumerate(row)])))
    return out

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