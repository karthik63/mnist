import numpy as np
from math import exp
import csv


lr = float(input("initial learning rate\n"))

batch_size = int(input("batch size\n"))

max_epochs = int(input("max epochs to run\n"))

act_fun = int(input("choice of activation function: 1 for tanh, 0 for sigmoid\n"))

num_hidden = int(input("number of hidden layers ?\n"))

layer_vals = np.zeros(num_hidden + 2)

layer_vals[0] = 784

for i in range (num_hidden):
    layer_vals[i+1]=input("size of hidden layer " + str(i+1)+'\n')

layer_vals[num_hidden + 1] = 10
layer_vals = layer_vals.astype(int)
layer_maxval = np.amax(layer_vals)
nn_activation = np.random.rand(num_hidden + 2, int(layer_maxval + 1))
nn_derivative_cost_wrt_s = np.random.rand(num_hidden + 2, int(layer_maxval + 1))
num_layers = int(num_hidden + 2)
nn_weights = np.random.uniform(-0.099, 0.099, (num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
nn_gradients = np.zeros((num_layers, int(layer_maxval + 1), int(layer_maxval + 1)))
nn_activation[0][784] = 1

'''
for b15 in range(0, layer_vals[1]):
    for b16 in range(0, layer_vals[0] + 1):
        nn_weights[1][b16][b15] *= .0001
'''

for i in range(0, num_layers):
    nn_activation[i][int(layer_vals[i])] = 1


def function(s):

    if act_fun == 0:

        if s > 300:
            return 1

        if s < -300:
            return 0

        return 1 / (1 + exp(-s))

    elif act_fun == 1:

        if s > 300:
            return 1

        if s < -300:
            return -1

        return (exp(s) - exp(-s)) / (exp(s) + exp(-s))


def predict(pic_bits):
    for i in range(0, 784):
        nn_activation[0][i] = pic_bits[i]

    for i in range(1, int(num_layers)):
        for j in range(0, int(layer_vals[i])):
            x = 0
            for k in range(0, int(layer_vals[i - 1]) + 1):
                x += nn_weights[i][k][j] * nn_activation[i - 1][k]

            nn_activation[i][j] = function(x)

    mx = 0
    mxindex = 0

    for i in range(0, 10):
        if nn_activation[num_layers - 1][i] > mx:
            mx = nn_activation[num_layers - 1][i]
            mxindex = i

    return mxindex


def derivative_activ_fun_wrt_s(s):

    if act_fun == 0:
        return function(s)*(1 - function(s))

    if act_fun == 1:
        return 1 - (function(s)) ** 2


def derivative_cost_wrt_x_final(a7, out_vector):
        return nn_activation[num_layers - 1][a7] - out_vector[a7]


def train(X, Y):

    net_loss = 0

    for a1 in range(0, max_epochs):

        num_points_seen = 0

        for a2 in range(0, X.shape[0]):

            nn_activation.fill(0)

            for i in range(0, num_layers):
                nn_activation[i][int(layer_vals[i])] = 1

            for a3 in range(0, 784):
                nn_activation[0][a3] = X[a2][a3]

            for a4 in range(1, num_layers):
                for a5 in range(0, layer_vals[a4]):
                    x = 0
                    for a6 in range(0, layer_vals[a4 - 1] + 1):
                        x += nn_activation[a4 - 1][a6] * nn_weights[a4][a6][a5]

                    nn_activation[a4][a5] = function(x)

            for a7 in range(0, 10):
                nn_derivative_cost_wrt_s[num_layers - 1][a7] = derivative_cost_wrt_x_final(a7, Y[
                    a2]) * derivative_activ_fun_wrt_s(nn_activation[num_layers - 1][a7])

            for a8 in range(num_layers - 2, 0, -1):
                for a9 in range(0, layer_vals[a8] + 1):
                    x = 0
                    for a10 in range(0, layer_vals[a8 + 1] + 1):

                        if a8 + 1 == num_layers - 1 and a10 == 10:
                            continue

                        x += nn_derivative_cost_wrt_s[a8 + 1][a10] * nn_weights[a8 + 1][a9][a10]

                    x *= derivative_activ_fun_wrt_s(nn_activation[a8][a9])
                    nn_derivative_cost_wrt_s[a8][a9] = x

            for a11 in range(1, num_layers):
                for a12 in range(0, layer_vals[a11]):
                    for a13 in range(0, layer_vals[a11 - 1] + 1):
                        nn_gradients[a11][a13][a12] += nn_activation[a11 - 1][a13] * nn_derivative_cost_wrt_s[a11][a12]

            num_points_seen += 1

            for a17 in range(0,10):
                net_loss += (Y[a2][a17] - nn_activation[num_layers - 1][a17])**2

            if num_points_seen % batch_size == 0:

                for a14 in range(1, num_layers):
                    for a15 in range(0, layer_vals[a14]):
                        for a16 in range(0, layer_vals[a14 - 1] + 1):
                            nn_weights[a14][a16][a15] -= lr * nn_gradients[a14][a16][a15]

                nn_gradients.fill(0)

                print("Epoch " + str(a1) + ", Step " + str(num_points_seen) + ", Loss: " + str(net_loss/100) + ", lr: " + str(lr))

                net_loss = 0


def test(X, Y):
    num_testcases = Y.shape[0]
    num_correct = 0

    for a1 in range(0, num_testcases):
        prediction = predict(X[a1])

        correct_prediction = -1

        for a2 in range(0, 10):
            if Y[a1][a2] == 1:
                correct_prediction = a2
                break

        if prediction == correct_prediction:
            num_correct += 1

    print("Accuracy is " + str(num_correct / num_testcases))


def main():
    raw_file = open('mnist_csv.csv', 'rt')
    reader = csv.reader(raw_file, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)
    mnist = np.array(x).astype('float')

    Y = mnist[:, 0]
    X = np.delete(mnist, 0, 1)

    Ynew = np.zeros(((Y.size), 10))

    for i in range(0, Y.size):
        Ynew[i][int(Y[i])] = 1

    Y = Ynew

    train(X,Y)

    raw_file = open('mnist_csv_test.csv', 'rt')
    reader = csv.reader(raw_file, delimiter=',', quoting=csv.QUOTE_NONE)
    xx = list(reader)
    mnist_ = np.array(xx).astype('float')

    Ytest = mnist_[:, 0]
    Xtest = np.delete(mnist_, 0, 1)

    Ynew_test = np.zeros(((Ytest.size), 10))

    for i in range(0, Ytest.size):
        Ynew_test[i][int(Ytest[i])] = 1

    Ytest = Ynew_test

    test(Xtest, Ytest)


main()


