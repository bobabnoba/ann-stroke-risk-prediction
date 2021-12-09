from __future__ import print_function
import numpy as np
from abc import abstractmethod
import math
import random
import copy
import pandas as pd
import sklearn
from tqdm import trange
import pickle
import keras
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


class ComputationalNode(object):

    @abstractmethod
    def forward(self, x):  # x is an array of scalars
        pass

    @abstractmethod
    def backward(self, dz):  # dz is a scalar
        pass


class MultiplyNode(ComputationalNode):

    def __init__(self):
        self.x = [0., 0.]  # x[0] is input, x[1] is weight

    def forward(self, x):
        self.x = x
        return self.x[0] * self.x[1]

    def backward(self, dz):
        return [dz * self.x[1], dz * self.x[0]]


class SumNode(ComputationalNode):

    def __init__(self):
        self.x = []  # x is in an array of inputs

    def forward(self, x):
        self.x = x
        return sum(self.x)

    def backward(self, dz):
        return [dz for xx in self.x]


class SigmoidNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._sigmoid(self.x)

    def backward(self, dz):
        return dz * self._sigmoid(self.x) * (1. - self._sigmoid(self.x))

    def _sigmoid(self, x):
        try:
            temp = 1. / (1. + math.exp(-x))
        except OverflowError:
            temp = float('inf')
        return temp


class TanhNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._tanh(self.x)

    def backward(self, dz):
        return dz * (1 - self._tanh(self.x ** 2))

    def _tanh(self, x):
        try:
            return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        except:
            return math.inf


class ReluNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._relu(self.x)

    def backward(self, dz):
        return dz * (1. if self.x > 0. else 0.)

    def _relu(self, x):
        return max(0., x)


class NeuronNode(ComputationalNode):

    def __init__(self, n_inputs, activation):
        self.n_inputs = n_inputs
        self.multiply_nodes = []  # for inputs and weights
        self.sum_node = SumNode()  # for sum of inputs*weights

        for n in range(n_inputs):  # collect inputs and corresponding weights
            mn = MultiplyNode()
            mn.x = [1., random.gauss(0., 0.1)]  # init input weights
            self.multiply_nodes.append(mn)

        mn = MultiplyNode()  # init bias node
        mn.x = [1., random.gauss(0., 0.01)]  # init bias weight
        self.multiply_nodes.append(mn)

        if activation == 'sigmoid':
            self.activation_node = SigmoidNode()
        elif activation == 'relu':
            self.activation_node = ReluNode()
        elif activation == 'tanh':
            self.activation_node = TanhNode()
        else:
            raise RuntimeError('Unknown activation function "{0}".'.format(activation))

        self.previous_deltas = [0.] * (self.n_inputs + 1)
        self.gradients = []

    def forward(self, x):  # x is a vector of inputs
        x = copy.copy(x)
        x.append(1.)  # for bias

        for_sum = []
        for i, xx in enumerate(x):
            inp = [x[i], self.multiply_nodes[i].x[1]]
            for_sum.append(self.multiply_nodes[i].forward(inp))

        summed = self.sum_node.forward(for_sum)
        summed_act = self.activation_node.forward(summed)
        return summed_act

    def backward(self, dz):
        dw = []
        dx = []
        b = dz[0] if type(dz[0]) == float else sum(dz)

        b = self.activation_node.backward(b)
        b = self.sum_node.backward(b)
        for i, bb in enumerate(b):
            dw.append(self.multiply_nodes[i].backward(bb)[1])
            dx.append(self.multiply_nodes[i].backward(bb)[0])

        self.gradients = dw
        return dx

    def update_weights(self, learning_rate, momentum):
        for i, multiply_node in enumerate(self.multiply_nodes):
            mean_gradient = self.gradients[i]
            delta = learning_rate * mean_gradient + momentum * self.previous_deltas[i]
            self.previous_deltas[i] = delta
            self.multiply_nodes[i].x[1] -= delta

        self.gradients = []


class NeuralLayer(ComputationalNode):

    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation

        self.neurons = []
        # construct layer
        for _ in range(n_neurons):
            neuron = NeuronNode(n_inputs, activation)
            self.neurons.append(neuron)

    def forward(self, x):  # x is a vector of "n_inputs" elements
        layer_output = []
        for neuron in self.neurons:
            neuron_output = neuron.forward(x)
            layer_output.append(neuron_output)

        return layer_output

    def backward(self, dz):  # dz is a vector of "n_neurons" elements
        b = []
        for idx, neuron in enumerate(self.neurons):
            neuron_dz = [d[idx] for d in dz]
            neuron_dz = neuron.backward(neuron_dz)
            b.append(neuron_dz[:-1])

        return b  # b is a vector of "n_neurons" elements

    def update_weights(self, learning_rate, momentum):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum)


class NeuralNetwork(ComputationalNode):

    def __init__(self):
        # construct neural network
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):  # x is a vector which is an input for neural net
        prev_layer_output = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:  # input layer
                prev_layer_output = layer.forward(x)
            else:
                prev_layer_output = layer.forward(prev_layer_output)

        return prev_layer_output  # actually an output from last layer

    def backward(self, dz):
        next_layer_dz = None
        for idx, layer in enumerate(self.layers[::-1]):
            if idx == 0:
                next_layer_dz = layer.backward(dz)
            else:
                next_layer_dz = layer.backward(next_layer_dz)

        return next_layer_dz

    def update_weights(self, learning_rate, momentum):
        for layer in self.layers:
            layer.update_weights(learning_rate, momentum)

    def fit(self, X, Y, learning_rate, momentum, nb_epochs, shuffle=False, verbose=0):
        assert len(X) == len(Y)

        hist = []
        for epoch in trange(nb_epochs):
            if shuffle:
                random.seed(epoch)
                random.shuffle(X)
                random.seed(epoch)
                random.shuffle(Y)

            total_loss = 0.0
            for x, y in zip(X, Y):
                # forward pass to compute output
                pred = self.forward(x)
                # compute loss
                grad = 0.0
                for o, t in zip(pred, y):
                    total_loss += (t - o) ** 2.
                    grad += -(t - o)
                # backward pass to compute gradients
                self.backward([[grad]])
                # update weights with computed gradients
                self.update_weights(learning_rate, momentum)

            hist.append(total_loss)
        if verbose == 1:
            print('Epoch {0}: loss {1}'.format(epoch + 1, total_loss))
        print('Loss: {0}'.format(total_loss))
        return hist

    def predict(self, x):
        return self.forward(x)


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add(NeuralLayer(18, 18, 'sigmoid'))
    nn.add(NeuralLayer(18, 15, 'tanh'))
    nn.add(NeuralLayer(15, 1, 'sigmoid'))

    D = pd.read_csv(r'dataset.csv')

    count_s0, count_s1 = D.stroke.value_counts()
    df_s0 = D[D['stroke'] == 0]
    df_s1 = D[D['stroke'] == 1]

    # df_s0_under = df_s0.sample(count_s1)
    # df_data_under = pd.concat([df_s0_under, df_s1], axis=0)
    # df_data_under.to_csv('dataset_undersample.csv', index=False)

    df_s1_over = df_s1.sample(count_s0, replace=True)
    df_data_over = pd.concat([df_s0, df_s1_over], axis=0)
    df_data_over.to_csv('dataset_oversample.csv', index=False)

    D = pd.read_csv(r'dataset_oversample.csv')
    print(D)
    print('-------------------------------------------')
    print('-------------------------------------------')

    # X = pd.read_csv(r'dataset_undersample.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Y = pd.read_csv(r'dataset_undersample.csv', usecols=[11])

    X = pd.read_csv(r'dataset_oversample.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Y = pd.read_csv(r'dataset_oversample.csv', usecols=[11])

    print(X)
    print('-------------------------------------------')
    print(Y)

    # X.avg_glucose_level = (X.avg_glucose_level - X.avg_glucose_level.min()) / (
    #         X.avg_glucose_level.max() - X.avg_glucose_level.min())
    # X.bmi = (X.bmi - X.bmi.min()) / (X.bmi.max() - X.bmi.min())
    # X.age = (X.age - X.age.min()) / (X.age.max() - X.age.min())
    # X.heart_disease = (X.heart_disease - X.heart_disease.min()) / (X.heart_disease.max() - X.heart_disease.min())
    # X.hypertension = (X.hypertension - X.hypertension.min()) / (X.hypertension.max() - X.hypertension.min())

    # O N E H O T  E N C O D I N G
    X = pd.concat([X, pd.get_dummies(X['gender'], prefix='gender', dummy_na=False)], axis=1).drop(['gender'], axis=1)
    X = pd.concat([X, pd.get_dummies(X['work_type'], prefix='work', dummy_na=False)], axis=1).drop(['work_type'],
                                                                                                   axis=1)
    # X = pd.concat([X, pd.get_dummies(X['Residence_type'], prefix='res', dummy_na=False)], axis=1).drop(
    #     ['Residence_type'], axis=1)
    X = pd.concat([X, pd.get_dummies(X['smoking_status'], prefix='smokes', dummy_na=False)], axis=1).drop(
        ['smoking_status'], axis=1)
    X = pd.concat([X, pd.get_dummies(X['ever_married'], prefix='married', dummy_na=False)], axis=1).drop(
        ['ever_married'], axis=1)

    # Missing values imputation; M E A N strategy
    X.bmi.fillna(X.bmi.mean(), inplace=True)

    X = X.drop(['smokes_Unknown'], axis=1)
    X = X.drop(['Residence_type'], axis=1)

    X['stroke'] = Y

    # N O R M A L I Z A T I O N
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(x_scaled)

    X.to_csv("final.csv", index=False)

    print('--------------n o r m a l i z o v a n o-----------------')
    N = pd.read_csv(r'final.csv')
    print(N)

    XX = pd.read_csv('final.csv',
                     usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    YY = pd.read_csv('final.csv', usecols=[18])

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(XX, YY, test_size=0.3, random_state=42)

    Xtrain = Xtrain.values.tolist()
    Ytrain = Ytrain.values.tolist()
    Xtest = Xtest.values.tolist()
    Ytest = Ytest.values.tolist()
    #history = nn.fit(Xtrain, Ytrain, learning_rate=0.1, momentum=0.9, nb_epochs=25, shuffle=True, verbose=0)

    # ****************** PICKLE ME ********************
    # with open('try2.pkl', 'wb') as f:
    #     pickle.dump(nn, f)
    #
    with open('bezRes.pkl', 'rb') as f:
        nn = pickle.load(f)

    ################### K E R A S #####################################################################
    # nn = Sequential()
    # nn.add(Dense(8, input_dim=20, activation='tanh'))
    # nn.add(Dense(8, activation='tanh'))
    # nn.add(Dense(1, activation='sigmoid'))
    #
    # nn.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    # nn.fit(Xtrain, Ytrain, epochs=25, batch_size=10)
    #
    # _, accuracy = nn.evaluate(Xtest, Ytest)
    # print('Accuracy: %.2f' % (accuracy * 100))
    ##################################################################################################

    acc = 0
    for i, j in zip(Xtest, Ytest):
        print('Predicted:', nn.predict(i), '\tExpected:', j)
        if -0.5 < j[0] - nn.predict(i)[0] < 0.5:
            acc += 1
    accuracy = (acc / len(Xtest)) * 100
    print('Accuracy: ', round(accuracy, 2), '%')
    y_pred = [nn.predict(x) for x in Xtest]
    # acc = sklearn.metrics.r2_score(Ytest, y_pred)
    # print('Accuracy(r2_score): ' + str(acc))

    # C O N F U S I O N  M A T R I X
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, j in zip(Xtest, Ytest):
        if int(round(nn.predict(i)[0])) == 1 and j[0] == 1:
            tp += 1
        elif int(round(nn.predict(i)[0])) == 0 and j[0] == 0:
            tn += 1
        elif int(round(nn.predict(i)[0])) == 1 and j[0] == 0:
            fp += 1
        elif int(round(nn.predict(i)[0])) == 0 and j[0] == 1:
            fn += 1

    print('True positive:', tp)
    print('True negative:', tn)
    print('False positive:', fp)
    print('False negative: ', fn)

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    F1_score = (precision * recall) / (precision + recall)

    tpr = tp / (tp + fn)

    tnr = tn / (tn + fp)

    balanced_acc = (tpr + tnr) / 2

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', F1_score)
    print('Balanced accuracy:', balanced_acc)

    # pyplot.plot(history)
    # pyplot.show()
