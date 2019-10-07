# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:11:13 2019

@author: Robin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:14:03 2019

@author: Robin
"""

__author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class DeepNeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_layer_num, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01,
                 seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        :param nn_layer_num: number of hidden layers
        '''
        self.nn_layer_num = nn_layer_num
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = [np.zeros((self.nn_input_dim, self.nn_hidden_dim))]
        self.W.append(np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim))
        self.b = [np.zeros((1, self.nn_hidden_dim))]
        self.b.append(np.zeros((1, self.nn_hidden_dim)))

        for i in range(nn_layer_num - 1):
            self.W.append(np.random.randn(self.nn_hidden_dim, self.nn_hidden_dim) / np.sqrt(self.nn_hidden_dim))
            self.b.append(np.zeros((1, self.nn_hidden_dim)))

        self.W.append(np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim))
        self.b.append(np.zeros((1, self.nn_output_dim)))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if type == 'tanh':
            actFunction = np.tanh(z)
        if type == 'sigmoid':
            actFunction = 1 / (1 + np.exp(-z))
        if type == 'relu':
            actFunction = np.maximum(0, z)

        return actFunction

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if type == 'tanh':
            diff_act = 1 - np.power(np.tanh(z), 2)
        if type == 'sigmoid':
            diff_act = np.exp(-z) / np.power((1 + np.exp(-z)), 2)
        if type == 'relu':
            diff_act = np.zeros(z.shape[0])
            diff_act[z > 0] = 1

        return diff_act

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        self.z = []
        self.z.append(X)
        self.a = []
        self.a.append(X)

        for i in range(1, self.nn_layer_num + 2):
            self.z.append(self.a[i - 1] @ self.W[i] + self.b[i])
            self.a.append(self.actFun(self.z[i], type=self.actFun_type))

        del (self.a[self.nn_layer_num + 1])

        exp_scores = np.exp(self.z[self.nn_layer_num + 1])
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE

        y_one_hot = np.zeros((num_examples, self.nn_output_dim))
        y_one_hot[range(0, num_examples), y] = 1

        data_loss = -np.sum(y_one_hot * np.log(self.probs))

        # Add regulatization term to loss (optional)
        add_loss = 0
        for i in range(1, self.nn_layer_num + 2):
            add_loss = add_loss + np.sum(np.square(self.W[i]))

        data_loss += self.reg_lambda / 2 * add_loss
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        delta3 = self.probs

        delta3[range(num_examples), y] -= 1

        dW = list(range(0, self.nn_layer_num + 2))

        dW[self.nn_layer_num + 1] = np.transpose(self.a[self.nn_layer_num]) @ delta3

        db = list(range(0, self.nn_layer_num + 2))

        db[self.nn_layer_num + 1] = np.sum(delta3, axis=0)

        delta_n = delta3

        for i in range(self.nn_layer_num, 0, -1):
            dW[i] = np.transpose(self.a[i - 1]) @ (
                        (delta_n @ np.transpose(self.W[i + 1])) * self.diff_actFun(self.z[i], self.actFun_type))

            delta_n = ((delta_n @ np.transpose(self.W[i + 1])) * self.diff_actFun(self.z[i], self.actFun_type))

            db[i] = np.sum(delta_n, axis=0)

        # dW[2] = np.transpose(self.a[1]) @ delta3

        # db[2] = np.sum(delta3, axis=0)

        # dW[1] = np.transpose(X) @ ((delta3 @ self.W[2].T) * self.diff_actFun(self.z[1], self.actFun_type))

        # db[1] = np.sum((delta3 @ self.W[2].T) * self.diff_actFun(self.z[1], self.actFun_type), axis=0)

        # dW2 = dL/dW2
        # db2 = dL/db2
        # dW1 = dL/dW1
        # db1 = dL/db1
        return dW, db

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW, db = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)

            for j in range(1, self.nn_layer_num + 2):
                dW[j] += self.reg_lambda * self.W[j]
                self.W[j] += -epsilon * dW[j]
                self.b[j] += -epsilon * db[j]

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    model = DeepNeuralNetwork(nn_layer_num=3, nn_input_dim=2, nn_hidden_dim=10, nn_output_dim=2, actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()