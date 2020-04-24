import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - tanh(z) * tanh(z)


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    return z > 0

import os
import numpy as np
import random


class NeuralNetwork(object):

    def __init__(
        self,
        sizes=[784, 30, 10],
        learning_rate=1e-2,
        mini_batch_size=16,
        activation_fn="relu"
    ):
        
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.activation_fn = getattr(activations, activation_fn)
        self.activation_fn_prime = getattr(activations, f"{activation_fn}_prime")

        
        self.weights = [np.array([0])] + [np.random.randn(y, x)/np.sqrt(x) for y, x in
                                          zip(sizes[1:], sizes[:-1])]

        
        self.biases = [np.array([0])] + [np.random.randn(y, 1) for y in sizes[1:]]

        self._zs = [np.zeros(bias.shape) for bias in self.biases]

      
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.lr = learning_rate

    def fit(self, training_data, validation_data=None, epochs=10):
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                for x, y in mini_batch:
                    self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [
                    w - (self.lr / self.mini_batch_size) * dw for w, dw in
                    zip(self.weights, nabla_w)]
                self.biases = [
                    b - (self.lr / self.mini_batch_size) * db for b, db in
                    zip(self.biases, nabla_b)]

            if validation_data:
                accuracy = self.validate(validation_data) / 100.0
                print(f"Epoch {epoch + 1}, accuracy {accuracy} %.")
            else:
                print(f"Processed epoch {epoch}.")

    def validate(self, validation_data):
        
        validation_results = [(self.predict(x) == y) for x, y in validation_data]
        return sum(result for result in validation_results)

    def predict(self, x):
       

        self._forward_prop(x)
        return np.argmax(self._activations[-1])

    def _forward_prop(self, x):
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self.weights[i].dot(self._activations[i - 1]) + self.biases[i]
            )
           
            if i == self.num_layers - 1:
                self._activations[i] = activations.softmax(self._zs[i])
            else:
                self._activations[i] = self.activation_fn(self._zs[i])

    def _back_prop(self, x, y):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - y)
        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error),
                self.activation_fn_prime(self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].transpose())

        return nabla_b, nabla_w

    def load(self, filename='model.npz'):
     
        npz_members = np.load(os.path.join(os.curdir, 'models', filename))

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])

     
        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

       
        self.mini_batch_size = int(npz_members['mini_batch_size'])
        self.lr = float(npz_members['lr'])

    def save(self, filename='model.npz'):
       
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=self.weights,
            biases=self.biases,
            mini_batch_size=self.mini_batch_size,
            lr=self.lr
        )

import gzip
import os
import pickle
import sys
import wget
import numpy as np
def load_mnist():
    if not os.path.exists(os.path.join(os.curdir, "data")):
        os.mkdir(os.path.join(os.curdir, "data"))
        wget.download("http://deeplearning.net/data/mnist/mnist.pkl.gz", out="data")

    data_file = gzip.open(os.path.join(os.curdir, "data", "mnist.pkl.gz"), "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [vectorized_result(y) for y in train_data[1]]
    train_data = list(zip(train_inputs, train_results))

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = val_data[1]
    val_data = list(zip(val_inputs, val_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))
    return train_data, val_data, test_data


def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e


if __name__ == "__main__":
    np.random.seed(42)

    layers = [784, 30, 10]
    learning_rate = 0.01
    mini_batch_size = 16
    epochs = 100

    # Initialize train, val and test data
    train_data, val_data, test_data = load_mnist()

    nn = NeuralNetwork(layers, learning_rate, mini_batch_size, "relu")
    nn.fit(train_data, val_data, epochs)

    accuracy = nn.validate(test_data) / 100.0
    print(f"Test Accuracy: {accuracy}%.")

    nn.save()