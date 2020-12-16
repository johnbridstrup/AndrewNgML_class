import numpy as np
from numpy import random
import matplotlib.pyplot as plt


## Generate a straight line with random m and random b

# Function generator for a straight line
def line(m,b):
    # define and return hypothesis function
    def y(x):
        try:
            return m*x + b
        except:
            return [m*xx + b for xx in x]
    return y

# Cost function and gradient
def cost(guesses, training, x):
    #cost
    cost = sum([(guess-known)**2 for guess,known in zip(guesses,training)])/(2*len(guesses))
    #partial wrt m
    m_deriv = sum([xi*(gi-ti) for xi,gi,ti in zip(x,guesses,training)])/len(guesses)
    #partial wrt b
    b_deriv = sum([(gi-ti) for gi,ti in zip(guesses,training)])/len(guesses)
    return cost, m_deriv, b_deriv


# Generate slope between -5 and 5
m = 10*random.rand()-5

# Generate intercept between -5 and 5
b = 10*random.rand()-5

# Get line function and populate data
initLine = line(m,b)
x=[i for i in range(100)]
cleanData = initLine(x)

# apply noise to data
mu, sigma = 0,100 # works with even very large sigma
y_data = [i + random.normal(mu,sigma) for i in cleanData]

# plt.figure()
# plt.plot(x,cleanData)
# plt.scatter(x,y_data)
# plt.show()
 
# make initial guesses
mg, bg = 0,0
h = line(mg,bg)

#Some parameters
max_iterations = 1000 # max number of iterations
curr_iter = 0
cost_threshold = 0.01 # end when Delta_cost/cost < threshold
alpha = 0.01 # learning rate

cost_hist =[]

prev_cost = 0
new_cost, m_deriv, b_deriv = cost(h(x), y_data, x)
print(cost(h(x),y_data,x))
while abs(new_cost-prev_cost)/new_cost > cost_threshold:
    curr_iter = curr_iter+1
    if curr_iter>max_iterations:
        raise Exception()
    prev_cost = new_cost
    mg = mg - alpha*m_deriv
    bg = bg - alpha*b_deriv
    h = line(mg,bg)
    new_cost, m_deriv, b_deriv = cost(h(x), y_data, x)
    cost_hist.append(new_cost)
    if new_cost > prev_cost:
        alpha = alpha*0.01
    print(cost(h(x),y_data,x))

plt.figure()
plt.plot(x, h(x), 'k')
plt.plot(x, initLine(x), '--', color='r')
plt.scatter(x, y_data)
plt.figure()
plt.plot(cost_hist)
plt.xlabel('iterations')
plt.ylabel('J')
plt.show()