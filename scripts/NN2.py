# code from https://github.com/erikdelange/Neural-networks-in-numpy/blob/master/network1.py

# my own neural network code did not learn : see /scripts/NN.py

# Fully configurable neural network in numpy

import numpy as np

np.set_printoptions(formatter={"float": "{: 0.3f}".format}, linewidth=np.inf)
np.random.seed(1)


class NN:
    def __init__(self, shape=(8,3,8)):
        """
        Initialize the neural network by creating empty arrays for values that will be 
        calculated later, and set initial weights and biases to small random values.
        Input: shape : a tuple with the number of nodes in each layer
        """
        self.num_layers = len(shape) - 1  # the input layer does not count as a layer
        self.weight = []
        self.bias = []

        self.a = []  # output of each layer
        self.z = []  # output of each layer before the activation function is applied
        self.dw = []  # weight correction matrix
        self.db = []  # bias correction matrix

        # initialize weights and biases and their corresponding update matrices to the appropriate sizes
        # give initial weights and biases small random values
        for (layer1, layer2) in zip(shape[:-1], shape[1:]):
            self.weight.append(np.random.normal(size=(layer2, layer1)))
            self.bias.append(np.random.normal(size=(layer2, 1)))
            self.dw.append(np.zeros((layer2, layer1)))
            self.db.append(np.zeros((layer2, 1)))

    def forward(self, x):
        """
        Feed-forward part of the algorithm in which a prediction is calculated
        Input: training data, a matrix of features and observations
        Output: matrix of predictions
        """
        self.a = [x.T]  # a[0] is the input for layer 1 (layer 0 is the input layer)
        self.z = [None]
        for (weight, bias) in zip(self.weight, self.bias):
            # apply weights and biases to the inputs of each layer
            self.z.append(weight.dot(self.a[-1]) + bias)
            # apply the activation function of choice (sigmoid) to find the output of each layer
            self.a.append(activation(self.z[-1]))

        return self.a[-1].T # prediction (output of last layer)

    def back_propagation(self, x, y, learning_rate=0.1, momentum=0.5):
        """
        Use weighted errors to adjust weights and biases
        Inputs: x : training data features and observations
            y: training data classification associated with each observations
            learning rate: factor controlling how much the weights change with each iteration
            momentum: factor controlling how much impact is given to the current values in the weight and 
                bias update matrices during adjustment
        Output: error: sum of the squared difference between predicted given outputs
        """
        # number of training examples
        m = x.shape[0]
        delta_w = []
        delta_b = []
        
        # perform forward propogation to calculated predicted values
        y_hat = self.forward(x)
        # sum of the squared difference between predicted given outputs
        error = np.sum((y_hat - y) ** 2)

        # iterate backwards through the layers
        for index in reversed(range(1, self.num_layers + 1)):
            # error for the output layer
            if index == self.num_layers:
                da = self.a[index] - y.T
            # error for the hidden layers
            else:
                da = self.weight[index].T.dot(dz)
            dz = da * activation(self.z[index], derivative=True)
            # compute gradient of the weights
            dw = dz.dot(self.a[index - 1].T) / m
            # compute gradient of the biases
            db = np.sum(dz, axis=1, keepdims=True) / m
            # store gradients to calculate updates
            delta_w.append(dw)
            delta_b.append(db)
        # iterate forward through layers to update weights and biases
        for (index, dw, db) in zip(reversed(range(self.num_layers)), delta_w, delta_b):
            # update the weight correction matrix
            self.dw[index] = learning_rate * dw + momentum * self.dw[index]
            # update weights
            self.weight[index] -= self.dw[index]
            # update the bias correction matrix
            self.db[index] = learning_rate * db + momentum * self.db[index]
            # update the biases
            self.bias[index] -= self.db[index]

        return error

    def train(self, x, y, iterations=10000, learning_rate=0.2, momentum=0.5, verbose=True):
        """
        Repeatedly predict, calculate error, and adjust weights accordingly to train a model according to a set of observations
        Inputs: x : training data features and observations
            y: training data classification associated with each observations
            iterations : the number of repetitions of updating weights
            learning rate: factor controlling how much the weights change with each iteration
            momentum: factor controlling how much impact is given to the current values in the weight and 
                bias update matrices during adjustment
            verbose: whether to print updates on error during the training process
        Outputs: loss : the sum of squared error over the course of the training process
        """
        min_error = 0.5
        loss = []
        for i in range(iterations + 1):
            error = self.back_propagation(x, y, learning_rate=learning_rate, momentum=momentum)
            loss.append(error)
            if verbose:
                if i % 2500 == 0:
                    print("iteration {:5d} error: {:0.6f}".format(i, error))
                # if error changes less than 0.5% in 100 trials, then stop
                if error <= min_error:
                    print("minimum error {} reached at iteration {}".format(min_error, i))
                    break
        return loss
        
def activation(x,derivative=False):
        """
        sigmoid activation function
        """
        if derivative:
            return (activation(x,False)*(1-activation(x,False)))
        else:
            return (1 / (1 + np.exp(-x)))
        