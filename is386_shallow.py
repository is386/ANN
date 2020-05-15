# Indervir Singh
# is386
# CS615
# HW 2
# Question 3


from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
from matplotlib.pyplot import plot, xlabel, ylabel, savefig, figure
from seaborn import set, heatmap
from pandas import DataFrame


YALE_PATH = "./yalefaces/"
IMAGE_X = 40
IMAGE_Y = 40
SEED = 11519991

# Hyper Parameters
LR = 0.001
TC = 10000
L2_RT = 0.001
H_NODES = 10
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


def theta_deriv(h, y, y_hat, theta):
    """
    Computes the gradient for the theta weights.
    Uses sigmoid activation and LLE objective.

    :param h: `numpy.ndarray` the hidden layer output
    :param y: `numpy.ndarray` the labels
    :param y_hat: `numpy.ndarray` the guess values
    :param theta: `numpy.ndarray` the hidden layer weights
    :return: `np.ndarray` the gradient for theta
    """
    l2 = 2 * (L2_RT * theta)
    return np.dot(h.T, (y - y_hat)) + l2


def beta_deriv(x, h, y, y_hat, theta, beta):
    """
    Computes the gradient for the theta weights.
    Uses sigmoid activation and LLE objective.

    :param x: `numpy.ndarray` the input layer data
    :param h: `numpy.ndarray` the hidden layer output
    :param y: `numpy.ndarray` the labels
    :param y_hat: `numpy.ndarray` the guess values
    :param theta: `numpy.ndarray` the hidden layer weights
    :param beta: `numpy.ndarray` the input layer weights
    :return: `np.ndarray` the gradient for theta
    """
    l2 = 2 * (L2_RT * beta)
    return np.dot(x.T, np.dot((y - y_hat), theta.T) * (h * (1 - h))) + l2


def train_network(x, y, theta, beta):
    """
    Trains the weights using batch gradient descent until the termination criteria is reached.

    :param x: `numpy.ndarray` the training data
    :param y: `numpy.ndarray` the training labels
    :param theta: `numpy.ndarray` untrained/randomized weights
    :param beta: `numpy.ndarray` untrained/randomized weights
    :return: `np.ndarray` the trained hidden weights
    :return: `np.ndarray` the trained input weights
    :return: `np.ndarray` the log likelihood errors
    """
    J = []
    for i in range(TC):
        # Forward for hidden
        net_h = np.dot(x, beta)
        h = sigmoid(net_h)

        # Forward for output
        net_y = np.dot(h, theta)
        y_hat = sigmoid(net_y)

        # Objective
        j = np.sum(log_like(y, y_hat))
        J.append(j)

        # Backward for theta
        theta_grad = theta_deriv(h, y, y_hat, theta)
        theta = theta + (LR * theta_grad)

        # Backward for beta
        beta_grad = beta_deriv(x, h, y, y_hat, theta, beta)
        beta = beta + (LR * beta_grad)

    return theta, beta, J


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


def test_network(x, y, theta, beta):
    """
    Tests the given test data with the trained weights.

    :param test_X: `numpy.ndarray` test data
    :param test_Y: `numpy.ndarray` test labels
    :param weights: `numpy.ndarray` trained weights
    :return: `float` the average
    :return: `numpy.ndarray` the confusion matrix
    """
    correct = 0
    size = y.shape[1]
    conf_mat = np.zeros((size, size))

    net_h = np.dot(x, beta)
    h = sigmoid(net_h)
    net_y = np.dot(h, theta)
    y_hat = sigmoid(net_y)

    for i in range(y.shape[0]):
        # gets the index where the label contains a 1. represents the subject
        actual = np.where(y[i] == 1)[0][0]
        # gets the index where the label contains a 1. represents most likely subject
        guess = np.where(y_hat[i] == np.amax(y_hat[i]))[0][0]
        conf_mat[guess][actual] += 1
        # get the index of the position that has a max, then add 2 to get the subject number
        if int(actual) + 2 == guess + 2:
            correct += 1

    return (correct / len(y)) * 100, conf_mat


def plot_j(J):
    """
    Plots the average log likelihood

    :param avg_J: `numpy.ndarray` test data
    """
    plot(range(len(J)), J)
    xlabel("Iterations")
    ylabel("Average Log Likelihood")
    savefig('log_like1.png', bbox_inches='tight')


def plot_conf_mat(mat):
    df_cm = DataFrame(mat, index=[i for i in range(
        2, 16)], columns=[i for i in range(2, 16)])
    figure(figsize=(10, 7))
    set(font_scale=1.2)
    heatmap(df_cm, annot=True)
    savefig('conf_mat1.png', bbox_inches='tight')


def main():
    np.random.seed(SEED)
    data_mat = parse_yale_faces()
    train_X, train_Y, test_X, test_Y = split_data(data_mat)
    standardize(train_X, test_X)
    train_labels = create_labels(train_Y)
    test_labels = create_labels(test_Y)

    input_size = train_X.shape[1]
    hidden_size = H_NODES
    output_size = train_labels.shape[1]
    beta = np.random.uniform(-0.0001, 0.0001, size=(input_size, hidden_size))
    theta = np.random.uniform(-0.0001, 0.0001, size=(hidden_size, output_size))

    theta, beta, J = train_network(train_X, train_labels, theta, beta)
    test_accuracy, conf_mat = test_network(test_X, test_labels, theta, beta)

    print("Testing Accuracy:", test_accuracy)
    print("Confusion Matrix:\n", conf_mat)

    plot_j(J)
    plot_conf_mat(conf_mat)


if __name__ == "__main__":
    main()
