from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

YALE_PATH = "./yalefaces/"
IMAGE_X = 40
IMAGE_Y = 40
SEED = 1151999

# Hyper Parameters
LR = 0.001
TC = 1000
L2_RT = 0.001
BIAS = 1

classes = {}


def parse_yale_faces():
    """
    Parses the yale faces data.

    :return: `np.ndarray` the image data
    """
    data_matrix = []
    yale_faces = [i for i in listdir(
        YALE_PATH) if isfile(join(YALE_PATH, i))]

    for face in yale_faces:
        try:
            face_img = Image.open(join(YALE_PATH, face))
            face_img = face_img.resize((IMAGE_X, IMAGE_Y))
            pixels = np.asarray(face_img).flatten()
            pixels = np.insert(pixels, 0, BIAS)
            face_img.close()
            sub_n = parse_subj_num(face)
            pixels = np.append(pixels, sub_n)
            data_matrix.append(pixels)

            # saves each class and its total
            if sub_n not in classes:
                classes[sub_n] = 1
            else:
                classes[sub_n] += 1
        except OSError:
            pass

    return np.asarray(data_matrix)


def split_data(data_mat):
    """
    Splits the overall data set into 2/3 training and 1/3 testing.

    :param data_mat: `np.ndarray` the data from the images
    :return: `np.ndarray` the training data
    :return: `np.ndarray` the training labels
    :return: `np.ndarray` the testing data
    :return: `np.ndarray` the testing labels
    """
    training = np.zeros((1, data_mat.shape[1]))
    testing = np.zeros((1, data_mat.shape[1]))

    for i in classes.keys():
        num_training = 2 * round(classes[i] / 3)
        # gets all of the data for each subject, i
        subj_mat = data_mat[data_mat[:, -1] == i]
        # shuffles that subject's data to reduce bias
        np.random.shuffle(subj_mat)
        # adds 2/3s of the data to training, and 1/3 to testing
        training = np.append(training, subj_mat[0:num_training, :], axis=0)
        testing = np.append(testing, subj_mat[num_training:, :], axis=0)

    # reshuffle to reduce bias
    np.random.shuffle(training)
    np.random.shuffle(testing)
    # labels are in the last column of the data
    return training[1:, :-1], training[1:, -1], testing[1:, :-1], testing[1:, -1]


def parse_subj_num(subject):
    """
    Parses the subject number from the subject name.

    :param subject: `str` the subject name
    :return: `int` the subject number
    """
    return int("".join(subject.split("subject")[1][0:2]))


def standardize(train_X, test_X):
    """
    Standardizes the training and test data.

    :param train_X: `numpy.ndarray` training data
    :param train_Y: `numpy.ndarray` test data
    """
    for i in range(1, train_X.shape[1]):
        s = np.std(train_X[:, i])
        m = np.mean(train_X[:, i])
        # if the std is 0, then set that feature to 0
        if s == 0:
            train_X[:, i] = 0
        train_X[:, i] = (train_X[:, i] - m) / s
        test_X[:, i] = (test_X[:, i] - m) / s


def create_labels(Y):
    """
    Takes the given labels and converts them to array of 0 and 1
    0 means it is not that class, and 1 means that is the correct class

    :param Y: `numpy.ndarray` labels
    :return: `numpy.ndarray` the multi class labels
    """
    labels = []
    for i in range(len(Y)):
        label = np.zeros((1, 15))
        # gets the subject number
        val = int(Y[i])
        # puts a one at the index representing the subject number
        label[0, val - 1] = 1
        labels.append(label)
    return np.asarray(labels).squeeze()[:, 1:]


def get_layers():
    """
    Gets the number of hidden layers and nodes from the user.

    :return: `np.ndarray` the hidden nodes
    """
    layer_nodes = []
    user_input = 0
    n = 0
    while True:
        try:
            user_input = int(input("Enter the number of layers: "))
            if user_input > 0:
                break
            else:
                print("Invalid input. Must be an integer greater than 0.\n")
        except ValueError:
            print("Invalid input. Must be an integer.\n")

    for i in range(user_input):
        while True:
            try:
                n = int(
                    input("Enter the number of nodes for layer %d: " % (i + 1)))
                if n > 0:
                    break
                else:
                    print("Invalid input. Must be an integer greater than 0.\n")
            except ValueError:
                print("Invalid input. Must be an integer.\n")
        layer_nodes.append(n)

    return layer_nodes


def init_weights(in_size, out_size, h_sizes):
    """
    Creates the network's layers and weights

    :param in_size: `int` size of the input layer
    :param out_size: `int` size of the output layer
    :param h_sizes: `list` list of the given hidden layer sizes
    :return: `np.ndarray` the weights for each layer
    """
    weights = []
    w0 = np.random.uniform(-0.0001, 0.0001, size=(in_size, h_sizes[0]))
    wn = np.random.uniform(-1, 1, size=(h_sizes[-1], out_size))
    weights.append(w0)

    for i in range(len(h_sizes) - 1):
        w = np.random.uniform(-1, 1, size=(h_sizes[i], h_sizes[i + 1]))
        weights.append(w)

    weights.append(wn)
    return weights


def sigmoid(z):
    """
    Computes the sigmoid function.

    :param z: `numpy.ndarray` dot product of weights and data
    :return: `np.ndarray` y_hat
    """
    return 1 / (1 + np.exp(-z))


def log_like(y, y_hat):
    """
    Computes the log likelihood.

    :param y: `numpy.ndarray` the label
    :param y_hat: `numpy.ndarray` the computation of the activation function
    :return: `np.ndarray` the log likelihoods
    """
    return (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def output_deriv(x, y, y_hat, weights):
    """
    Computes the gradient for log likelihood.

    :param x: `numpy.ndarray` the hidden layer output
    :param y: `numpy.ndarray` the labels
    :param y_hat: `numpy.ndarray` the guess values
    :param weights: `numpy.ndarray` the weights
    :return: `np.ndarray` the gradient for theta
    """
    l2 = 2 * (L2_RT * weights)
    return np.dot(x.T, (y - y_hat)) + l2


def back_prop(x, h_vals, y, y_hat, weights):
    """
    Computes the gradient for the theta weights.
    Uses sigmoid activation and LLE objective.

    :param x: `numpy.ndarray` the input layer data
    :param h_vals: `list` the hidden layer outputs
    :param y: `numpy.ndarray` the labels
    :param y_hat: `numpy.ndarray` the guess values
    :param weights: `numpy.ndarray` the weights for all the hidden layers
    :return: `np.ndarray` the gradient
    """
    rev_h_vals = h_vals[::-1] + [x]
    rev_weights = weights[::-1]

    for i in range(len(rev_weights[1:])):
        # Outer derivative
        grad = y - y_hat
        l2 = 2 * (L2_RT * rev_weights[i + 1])

        # Derivative of each layer until the current layer
        for j in range(i + 1):
            h = rev_h_vals[j]
            d = h * (1 - h)
            w = rev_weights[j]
            grad = np.dot(grad, w.T)
            grad = grad * d

        # Gets the current layer's input value
        h = rev_h_vals[i + 1]
        grad = np.dot(h.T, grad) + l2
        rev_weights[i + 1] = rev_weights[i + 1] + (LR * grad)

    return rev_weights[::-1]


def train_network(x, y, weights):
    """
    Trains the weights until the termination criteria is reached.

    :param x: `numpy.ndarray` the training data
    :param y: `numpy.ndarray` the training labels
    :param weights: `numpy.ndarray` untrained/randomized weights for each layer
    :return: `np.ndarray` the trained weights
    :return: `np.ndarray` the log likelihood errors
    """
    J = []
    for _ in range(TC):
        h_vals = []

        # Forward for input layer
        net_h = np.dot(x, weights[0])
        h = sigmoid(net_h)
        h_vals.append(h)

        # Forward for hidden layers
        for w in weights[1:-1]:
            net_h = np.dot(h, w)
            h = sigmoid(net_h)
            h_vals.append(h)

        # Forward for output layer
        net_y = np.dot(h_vals[-1], weights[-1])
        y_hat = sigmoid(net_y)

        # Objective
        j = np.mean(log_like(y, y_hat))
        J.append(j)

        # Backward for output layer
        out_grad = output_deriv(h, y, y_hat, weights[-1])
        weights[-1] = weights[-1] + (LR * out_grad)

        weights = back_prop(x, h_vals, y, y_hat, weights)

    return weights, np.asarray(J)


def test_network(x, y, weights):
    """
    Tests the given test data with the trained weights.

    :param test_X: `numpy.ndarray` test data
    :param test_Y: `numpy.ndarray` test labels
    :param weights: `numpy.ndarray` trained weights
    :return: `float` the average
    """
    correct = 0
    confuse_mat = np.zeros((len(classes), len(classes)))

    # Forward for input layer
    net_h = np.dot(x, weights[0])
    h = sigmoid(net_h)

    # Forward for hidden layers
    for w in weights[1:-1]:
        net_h = np.dot(h, w)
        h = sigmoid(net_h)

    # Forward for output layer
    net_y = np.dot(h, weights[-1])
    y_hat = sigmoid(net_y)

    for i in range(y.shape[0]):
        # gets the index where the label contains a 1. represents the subject
        actual = np.where(y[i] == 1)[0][0]
        # gets the index where the label contains a 1. represents most likely subject
        guess = np.where(y_hat[i] == np.amax(y_hat[i]))[0][0]
        confuse_mat[guess][actual] += 1

        # get the index of the position that has a max, then add 2 to get the subject number
        if int(actual) + 2 == guess + 2:
            correct += 1

    return (correct / len(y)) * 100, confuse_mat


def plot_avg_j(avg_J):
    """
    Plots the average log likelihood

    :param avg_J: `numpy.ndarray` test data
    """
    plt.plot(range(len(avg_J)), avg_J)
    plt.xlabel("Iterations")
    plt.ylabel("Average Log Likelihood")
    plt.savefig('log.png', bbox_inches='tight')
    plt.close()


def plot_confuse(confuse_mat):
    """
    Plots the confusion matrix

    :param confuse_mat: `numpy.ndarray` confusion matrix
    """
    df_cm = pd.DataFrame(
        confuse_mat,
        index=[i for i in range(2, 16)],
        columns=[i for i in range(2, 16)]
    )
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confuse.png', bbox_inches='tight')
    plt.close()


def main():
    np.random.seed(SEED)
    data_mat = parse_yale_faces()
    train_X, train_Y, test_X, test_Y = split_data(data_mat)
    standardize(train_X, test_X)
    train_labels = create_labels(train_Y)
    test_labels = create_labels(test_Y)

    h_sizes = get_layers()
    input_size = train_X.shape[1]
    output_size = train_labels.shape[1]
    weights = init_weights(input_size, output_size, h_sizes)

    weights, avg_j = train_network(train_X, train_labels, weights)
    test_acc, confuse_mat = test_network(test_X, test_labels, weights)
    plot_avg_j(avg_j)
    plot_confuse(confuse_mat)
    print("Testing Accuracy:", test_acc)


if __name__ == "__main__":
    main()
