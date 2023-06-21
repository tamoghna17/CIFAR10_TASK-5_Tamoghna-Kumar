import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os




def sigmoid(x):
  return 1/(1+np.exp(-x))

def tanh(x):
  return np.tanh(x)

def sigmoid_derivative(x):
    return (sigmoid(x))*(1-sigmoid(x))

def squared_error(y_true, y_pred):
    return np.sum(np.dot(y_pred-y_true, (y_pred-y_true).T), axis=0, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))

def gradient_descent(weight, bias, dweight, dbias, lr):
    for i in range(len(weight)):
        weight[i] = weight[i] - lr * dweight[i]
        bias[i] = bias[i] - lr * dbias[i]

def momentum(weight, bias, dweight, dbias, lr, momentum, prev_dweight, prev_dbias):
    for i in range(len(weights)):
        delta_weight = momentum * prev_delta_weight[i] - lr * dweight[i]
        delta_bias = momentum * prev_delta_bias[i] - lr * dbias[i]
        weight[i] = weight[i] + delta_weight
        bias[i] =bias[i] + delta_bias
        prev_delta_weight[i] = delta_weight
        prev_delta_bias[i] = delta_bias

def nag(weight, bias, dweight, dbias, lr, momentum, prev_delta_weight, prev_delta_bias):
    for i in range(len(weight)):
        delta_weight = momentum * prev_delta_weight[i]
        delta_bias = momentum * prev_delta_bias[i]
        weight[i] += delta_weight
        bias[i] += delta_bias
        delta_weight = delta_weight - lr * dweight[i]
        delta_bias = delta_bias - lr * dbias[i]
        weight[i] -= delta_weight
        bias[i] -= delta_bias
        prev_delta_weight[i] = delta_weight
        prev_delta_bias[i] = delta_bias

def adam(weight, bias, dweight, dbias, lr, beta1, beta2, epsilon, m, v, t):
    for i in range(len(weights)):
        m[i] = beta1 * m[i] + (1 - beta1) * dweight[i]
        v[i] = beta2 * v[i] + (1 - beta2) * dweight[i]**2
        m_hat = m[i] / (1 - beta1**t)
        v_hat = v[i] / (1 - beta2**t)
        weight[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

class nnlayer:
    def __init__(self, input_shape, output_shape, activation, opt, momentum, lr):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.opt = opt
        self.momentum = momentum
        self.lr = lr
        self.weight = np.random.randn(input_shape, output_shape)
        self.bias = np.zeros((output_shape, 1))
        if activation == 'sigmoid':
            self.activation = sigmoid
        elif activation == 'tanh':
            self.activation = tanh
        self.vweight = np.zeros_like(self.weight)
        self.vbias = np.zeros_like(self.bias)

    def forward(self, X):
        self.input = X
        self.z = np.dot(self.weight.T, X) + self.bias
        self.output = self.activation(self.z)
        return self.output

    def backward(self, delta, lr):
        if self.activation == sigmoid:
            delta *= sigmoid_derivative(self.z)
        elif self.activation == tanh:
            delta *= 1 - np.square(tanh(self.z))

        self.dweight = np.dot(delta, self.input.T).T / self.input.shape[0]
        self.dbias = np.mean(delta, axis=1)
        self.dbias = np.mean(self.dbias, axis=0)

        if self.opt == "gd":
            delta = np.dot(self.weight, delta)
            self.weight -= lr * self.dweight
            self.bias -= lr * self.dbias
            return delta
        elif self.opt == "momentum":
            self.vweight = self.momentum * self.vweight - self.lr * self.dweight
            self.vbias = self.momentum * self.vbias - self.lr * self.dbias
            self.weight += self.vweight
            self.bias += self.vbias

            delta = np.dot(delta, self.weights.T)
            return delta
        elif self.opt == "nag":
            vweight_prev = self.vweight
            vbias_prev = self.vbias
            self.vweight = self.momentum * self.vweight - self.lr * self.dweight
            self.vbias = self.momentum * self.vbias - self.lr * self.dbias

            self.weight += -self.momentum * vweight_prev + (1 + self.momentum) * self.vweight
            self.bias += -self.momentum * vbias_prev + (1 + self.momentum) * self.vbias

            delta = np.dot(delta, self.weight.T)
            return delta
        elif self.opt == "adam":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.mweight = np.zeros_like(self.weight)
            self.mbias = np.zeros_like(self.bias)
            self.vweight = np.zeros_like(self.weight)
            self.vbias = np.zeros_like(self.bias)
            self.t = 0

            self.t += 1
            self.mweight = self.beta1 * self.mweight + (1 - self.beta1) * self.dweight
            self.mbias = self.beta1 * self.mbias + (1 - self.beta1) * self.dbias
            self.vweight = self.beta2 * self.vweight + (1 - self.beta2) * np.square(self.dweight)
            self.vbias = self.beta2 * self.vbias + (1 - self.beta2) * np.square(self.dbias)

            mweight_corrected = self.mweight / (1 - self.beta1 ** self.t)
            mbias_corrected = self.mbias / (1 - self.beta1 ** self.t)
            vweight_corrected = self.vweight / (1 - self.beta2 ** self.t)
            vbias_corrected = self.vbias / (1 - self.beta2 ** self.t)

            self.weight -= self.lr * mweight_corrected / (np.sqrt(vweight_corrected) + self.epsilon)
            self.bias -= self.lr * mbias_corrected / (np.sqrt(vbias_corrected) + self.epsilon)

            delta = np.dot(delta, self.weight.T)
            return delta

class Neural_Network:
    def __init__(self, momentum, num_hidden, sizes, activation, lr, loss, opt):
        self.momentum = momentum
        self.hidden_layers = num_hidden
        self.sizes = sizes
        self.activation = activation
        self.layers = []
        self.lr = lr
        if loss=="ce":
            self.loss = cross_entropy_loss
        elif loss == "sq":
            self.loss = squared_error

        layer_input_shape = 32*32*3
        for i in range(self.hidden_layers):
            layer_output_shape = sizes[i]
            layer = nnlayer(layer_input_shape, layer_output_shape, activation, opt, momentum, lr)
            self.layers.append(layer)
            layer_input_shape = layer_output_shape
        output_layer = nnlayer(layer_input_shape, 10, activation, opt, momentum, lr)
        self.layers.append(output_layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward(self, delta, lr):
        for layer in reversed(self.layers):
            delta = layer.backward(delta, lr)
    
    def train(self, X_train, y_train, epochs, batch_size, anneal=False, X_val=None, y_val=None):
        num_samples = X_train.shape[0]
        num_steps = num_samples // batch_size

        for epoch in range(epochs):
            if anneal and epoch > 0 and epoch % 2 == 0:
                self.lr /= 2

            permutation = np.random.permutation(num_samples)
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            for step in range(num_steps):
                start = step * batch_size
                end = start + batch_size
                Xbatch = ((X_train.T)[start:end]).T
                ybatch = (y_train[start:end]).T

                y_pred = self.forward(Xbatch)
                loss = self.loss(y_pred, ybatch)
                error_rate = self.compute_error_rate(y_pred, ybatch)

                delta = y_pred - ybatch
                self.backward(delta, self.lr)

                if (step + 1) % 100 == 0:
                    print("Epoch ",epoch, "Step ", (step + 1),"Loss: ", loss, "Error: " ,error_rate, "lr: " ,self.lr)


            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss(y_val_pred, y_val)
                val_error_rate = self.compute_error_rate(y_val_pred, y_val)
                file = open("log train.txt", 'a')
                file.write("Epoch ",epoch, "Validation Loss: " ,val_loss, "Validation Error: ", val_error_rate,"\n")
                file.close()
            
            

    def compute_error_rate(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return np.mean(y_pred != y_true)
    
    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "model.pkl"), "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(save_dir):
        with open(os.path.join(save_dir, "model.pkl"), "rb") as file:
            return pickle.load(file)

    
