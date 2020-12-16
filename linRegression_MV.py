import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
    Multivariable Linear regression script

    Data on cost of insurance from https://github.com/stedy/Machine-Learning-with-R-datasets

    Data:
        Age: Int
        Sex: Category - M/F
        BMI: Float
        Children: Int
        Smoker: Bool
        Region: Category - northeast, northwest, southeast, southwest
        Charges: Float
"""

def formatData(data):
    """Format data with boolean values for categorical data"""
    # Binary data first
    binData = data.replace({'female':1, 'male':0, 'yes':1, 'no':0})
    # Category data
    regionDummies = binData.region.str.get_dummies()
    outData = pd.concat([regionDummies,binData.drop('region', axis=1)],axis=1)
    # Normalize float data (min/max)
    cols_to_norm = ['age','bmi']
    outData[cols_to_norm] = MinMaxScaler().fit_transform(outData[cols_to_norm])
    # add offset feature (identical to 1)
    outData.insert(0, 'offset', 1)
    return outData

def costGradient(guesses, training, x, features):
    # number of samples
    samples = len(x.index)

    # cost
    cost = sum([(guess-known)**2 for guess,known in zip(guesses,training)])/(2*len(guesses))/(2*samples)

    # Partial derivatives
    dTheta = []
    for feature in features:
        dth = sum([(guess - act)*row[feature] for guess, act, (index, row) in zip(guesses, training, x.iterrows())])/samples
        dTheta.append(dth)
    return cost, dTheta

def linFit(theta, features):
    def prediction(x):
        return [sum([th*row[ft] for th,ft in zip(theta,features)]) for index,row in x.iterrows()]
    return prediction

# Load data
path = 'data/insurance.csv'
data = pd.read_csv(path)
# print(data)

## Format data
formatted_data = formatData(data)
# print(formatted_data)
""" Sex and Smoker can be treated as a binary variable [0,1]. 
    Region converted into 4 boolean variables for NE, NW, SE, SW
"""
# Feature labels
features = [
    'offset',
    'northeast',
    'northwest',
    'southeast',
    'southwest',
    'age',
    'sex',
    'bmi',
    'children',
    'smoker'
] # Setup for regression
theta = [0 for _ in features]
max_iterations = 50 # max number of iterations
curr_iter = 0
cost_threshold = 0.01 # end when Delta_cost/cost < threshold
alpha = 0.5 # learning rate

cost_hist =[]

prev_cost = 0
fit_line = linFit(theta, features)
init_guess = fit_line(formatted_data)
training_data = formatted_data['charges'].tolist()
new_cost, dTheta = costGradient(init_guess, training_data, formatted_data, features)
cost_hist.append(new_cost)
while abs(new_cost-prev_cost)/new_cost > cost_threshold:
    curr_iter = curr_iter+1
    if curr_iter>max_iterations:
        break
    prev_cost = new_cost
    prev_theta = theta
    theta = [th - alpha*dth for th,dth in zip(prev_theta, dTheta)]
    fit_line = linFit(theta, features)
    guess = fit_line(formatted_data)
    new_cost, dTheta = costGradient(guess, training_data, formatted_data, features)
    cost_hist.append(new_cost)

print(theta)
plt.figure()
plt.plot(cost_hist)
plt.show()

