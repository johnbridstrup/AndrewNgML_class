import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1-sigmoid(z))

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

def nnCostFunction(nn_params,
                input_layer_size,
                hidden_layer_size,
                num_labels,
                X, y, lamb=0.0):
    """
    Implements the neural network cost function and gradient for a two layer neural 
    network which performs classification. 
    
    Parameters
    ----------
    nn_params : array_like
        The parameters for the neural network which are "unrolled" into 
        a vector. This needs to be converted back into the weight matrices Theta1
        and Theta2.
    
    input_layer_size : int
        Number of features for the input layer. 
    
    hidden_layer_size : int
        Number of hidden units in the second layer.
    
    num_labels : int
        Total number of labels, or equivalently number of units in output layer. 
    
    X : array_like
        Input dataset. A matrix of shape (m x input_layer_size).
    
    y : array_like
        Dataset labels. A vector of shape (m,).
    
    lambda_ : float, optional
        Regularization parameter.

    Returns
    -------
    J : float
        The computed value for the cost function at the current weight values.
    
    grad : array_like
        An "unrolled" vector of the partial derivatives of the concatenatation of
        neural network weights Theta1 and Theta2.
    
    Instructions
    ------------
    You should complete the code by working through the following parts.
    
    - Part 1: Feedforward the neural network and return the cost in the 
            variable J. After implementing Part 1, you can verify that your
            cost function computation is correct by verifying the cost
            computed in the following cell.
    
    - Part 2: Implement the backpropagation algorithm to compute the gradients
            Theta1_grad and Theta2_grad. You should return the partial derivatives of
            the cost function with respect to Theta1 and Theta2 in Theta1_grad and
            Theta2_grad, respectively. After implementing Part 2, you can check
            that your implementation is correct by running checkNNGradients provided
            in the utils.py module.
    
            Note: The vector y passed into the function is a vector of labels
                    containing values from 0..K-1. You need to map this vector into a 
                    binary vector of 1's and 0's to be used with the neural network
                    cost function.
    
            Hint: We recommend implementing backpropagation using a for-loop
                    over the training examples if you are implementing it for the 
                    first time.
    
    - Part 3: Implement regularization with the cost function and gradients.
    
            Hint: You can implement this around the code for
                    backpropagation. That is, you can compute the gradients for
                    the regularization separately and then add them to Theta1_grad
                    and Theta2_grad from Part 2.
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = y.size
        
    # You need to return the following variables correctly 
    J = 0

    # ====================== YOUR CODE HERE ======================
    if X.ndim == 1:
        one_d = True
        X = np.expand_dims(X, axis=0)
    elif X.ndim > 2 or X.ndim==0:
        raise IndexError('Must be 1D or 2D')

    # Make matrix of number vectors

    #### MY WAY DOESNT WORK FOR THE WEIGHTS GIVEN
    # y_mat = np.zeros((m,num_labels))
    # for idx in range(num_labels):
    #     num_vec = np.zeros(num_labels)
    #     num_vec[idx] = 1
    #     y_mat[idx] = y_mat[idx] + num_vec
    ####
    y_mat = y.reshape(-1)
    y_mat = np.eye(num_labels)[y_mat]
    
    # I no longer assume 1's have been added before input
    layer1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    z1 = layer1.dot(Theta1.T)
    layer2 = sigmoid(z1)
    layer2 = np.concatenate([np.ones((layer2.shape[0], 1)), layer2], axis=1)
    z2 = layer2.dot(Theta2.T)
    layer3 = sigmoid(z2) # This outputs predictions

    # Standard cost
    J = -(1/m)*np.sum((np.log(layer3)*y_mat)+np.log(1-layer3)*(1-y_mat))
    
    # Regularization
    t1 = Theta1
    t2 = Theta2
    reg = (lamb/(2*m)) * (np.sum(np.square(t1[:,1:])) + np.sum(np.square(t2[:,1:])))
    J = J + reg

    return J

def nnCostGradient(nn_params,
                input_layer_size,
                hidden_layer_size,
                num_labels,
                X, y, lamb=0.0, debug=False):
    
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    m = y.size
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    if debug:
        print('nnParam size: ', nn_params.size)
        print(' t1 and t2 ravel size: ', np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()]).size)
    if X.ndim == 1:
        one_d = True
        X = np.expand_dims(X, axis=0)
    elif X.ndim > 2 or X.ndim==0:
        raise IndexError('Must be 1D or 2D')

    # Make matrix of number vectors
    y_mat = y.reshape(-1)
    y_mat = np.eye(num_labels)[y_mat]
    
    layer1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    z1 = layer1.dot(Theta1.T)
    layer2 = sigmoid(z1)
    layer2 = np.concatenate([np.ones((layer2.shape[0], 1)), layer2], axis=1)
    # add_ones = np.ones((X.shape[0],1))
    # layer2 = np.hstack((add_ones, layer2))
    z2 = layer2.dot(Theta2.T)
    layer3 = sigmoid(z2) # This outputs predictions

    # deltas
    d3 = layer3 - y_mat
    sigGrad = sigmoidGradient(layer1.dot(Theta1.T))
    sigGrad = np.concatenate([np.ones((m, 1)), sigGrad], axis=1)
    d2 = d3.dot(Theta2) * sigGrad
    D1 = d2[:,1:].T.dot(layer1)
    D2 = d3.T.dot(layer2)
    # print('D1 and D2 ravel size: ', np.concatenate([D1.ravel(), D2.ravel()]).size)

    Theta1_grad = (1/m) * D1
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (lamb/m)*Theta1[:,1:]

    Theta2_grad = (1/m) * D2
    Theta2_grad[:, 1:] = Theta2_grad[:,1:] + (lamb/m)*Theta2[:,1:]

    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    if debug:
        print("Gradient size: ", grad.size)
    return grad

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    """
    Randomly initialize the weights of a layer in a neural network.
    
    Parameters
    ----------
    L_in : int
        Number of incomming connections.
    
    L_out : int
        Number of outgoing connections. 
    
    epsilon_init : float, optional
        Range of values which the weight can take from a uniform 
        distribution.
    
    Returns
    -------
    W : array_like
        The weight initialiatized to random values.  Note that W should
        be set to a matrix of size(L_out, 1 + L_in) as
        the first column of W handles the "bias" terms.
        
    Instructions
    ------------
    Initialize W randomly so that we break the symmetry while training
    the neural network. Note that the first column of W corresponds 
    to the parameters for the bias unit.
    """

    # You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================

    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    # ============================================================
    return W

def debugInitializeWeights(fan_out, fan_in):
    """
    Initialize the weights of a layer with fan_in incoming connections and fan_out outgoings
    connections using a fixed strategy. This will help you later in debugging.
    Note that W should be set a matrix of size (1+fan_in, fan_out) as the first row of W handles
    the "bias" terms.
    Parameters
    ----------
    fan_out : int
        The number of outgoing connections.
    fan_in : int
        The number of incoming connections.
    Returns
    -------
    W : array_like (1+fan_in, fan_out)
        The initialized weights array given the dimensions.
    """
    # Initialize W using "sin". This ensures that W is always of the same values and will be
    # useful for debugging
    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W

def computeNumericalGradient(J, theta, e=1e-4):
    """
    Computes the gradient using "finite differences" and gives us a numerical estimate of the
    gradient.
    Parameters
    ----------
    J : func
        The cost function which will be used to estimate its numerical gradient.
    theta : array_like
        The one dimensional unrolled network parameters. The numerical gradient is computed at
        those given parameters.
    e : float (optional)
        The value to use for epsilon for computing the finite difference.
    Notes
    -----
    The following code implements numerical gradient checking, and
    returns the numerical gradient. It sets `numgrad[i]` to (a numerical
    approximation of) the partial derivative of J with respect to the
    i-th input argument, evaluated at theta. (i.e., `numgrad[i]` should
    be the (approximately) the partial derivative of J with respect
    to theta[i].)
    """
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1 = J(theta - perturb[:, i])
        loss2 = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad

def checkNNGradients(gradientFunction, costFunction, lamb=0.0):
    """
    Creates a small neural network to check the backpropagation gradients. It will output the
    analytical gradients produced by your backprop code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient computations should result in
    very similar values.
    Parameters
    ----------
    nnCostFunction : func
        A reference to the cost function implemented by the student.
    lambda_ : float (optional)
        The regularization parameter value.
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = np.arange(1, 1+m) % num_labels
    # print(y)
    # Unroll parameters
    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

    # short hand for cost function
    costFunc = lambda p: costFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, lamb)
    gradFunc = lambda p: gradientFunction(p, input_layer_size, hidden_layer_size,
                                        num_labels, X, y, lamb)
    grad = gradFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.The two columns you get should be very similar.
    print(np.stack([numgrad, grad], axis=1))
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
        'the relative difference will be small (less than 1e-9). \n'
        'Relative Difference: %g' % diff)

def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p
