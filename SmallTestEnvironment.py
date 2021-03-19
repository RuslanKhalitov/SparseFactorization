"""
An implementation of a small 1-layer model.
Testing environment

"""

# Imports
import numpy as np
import math
from ChordMatrix import chord_mask

# Globals
N = 3  # Sequence length
d = 2  # Embedding size
parameters = {
    'f': {},
    'g': {}
}
LRg = 0.01
LRf = 0.01

def relu(x):
    return max(0, x)

def leaky_relu(x, slope=0.01):
    return max(slope * x, x)

activation_g = relu
activation_f = relu



def get_X(N, embed_size):
    """
    Generates X
    :param N: Sequence length
    :param embed_size: Embedding size
    :return: matrix Nxd
    """
    return np.random.randn(N, embed_size)


def get_Vgt(X, transform='exp'):
    """
    X -> V_gt
    :param X: The initial embedding (X)
    :param transform: An element-wise transformation applied to the input data
    :return: matrix Nxd
    """
    transform_to_func = {
        'exp': np.exp,
        'log': np.log
    }
    assert transform in transform_to_func.keys(), f"Unknown transformation {transform}," \
                                                  f" available: {transform_to_func.keys()}"
    return transform_to_func[transform](X)


def init_f(N, d):
    """
    Initializes f (d -> N)
    :param N: Sequence length
    :param d: Embedding size
    :return: None
    """
    weights = np.zeros((N, d))
    bias = np.zeros(N)
    grad_weights = np.zeros((N, N, d))
    grad_bias = np.zeros(N, N)

    parameters['f']['weights'] = weights
    parameters['f']['bias'] = bias
    parameters['f']['grad_weights'] = grad_weights
    parameters['f']['grad_bias'] = grad_bias


def init_g(N, d):
    """
    Initializes g (d -> d)
    :param N: Sequence length
    :param d: Embedding size
    :return: None
    """
    weights = np.zeros((d, d))
    bias = np.zeros(d)
    grad_weights = np.zeros((N, d, d))
    grad_bias = np.zeros(N, d)

    parameters['g']['weights'] = weights
    parameters['g']['bias'] = bias
    parameters['g']['grad_weights'] = grad_weights
    parameters['g']['grad_bias'] = grad_bias


def calculate_loss(V_gt, V0):
    """
    Calculates the loss value as MSE(V_gt, V0)
    :param V_gt: Ground truth after a transformation applied
    :param V0: Result of the forward process
    :return: Loss value
    """
    assert np.shape(V_gt) == np.shape(V0), "Input tensors have different size"
    loss = np.square(np.subtract(V_gt, V0)).mean()
    return loss


def forward(X, parameters):
    """
    Forward pass X -> V0
    :param parameters: Current parameters of f and g
    :return: V0
    """

    # V
    V = np.zeros(N, d)
    assert np.shape(V) == np.shape(X), "V and X have different size"
    xi = parameters['g']['weights']
    bias = parameters['g']['bias']

    # rows
    for i in range(N):
        # columns
        for j in range(d):
            # bias[j] + X[i, 0] * xi[j, 0] + X[i, 1] * xi[j, 1]
            V[i, j] = activation_g(bias[j] + np.sum([X[i, _] * xi[j, _] for _ in range(d)]))

    # W
    W_unmask = np.zeros(N, N)
    theta = parameters['f']['weights']
    bias = parameters['f']['bias']

    # rows
    for i in range(N):
        # columns
        for j in range(N):
            # bias[j] + X[i, 0] * theta[j, 0] + X[i, 1] * theta[j, 1] + X[i, 2] * theta[j, 2]
            W_unmask[i, j] = activation_f(bias[j] + np.sum([X[i, _] * theta[j, _] for _ in range(N)]))

    # masking W (elementwise)
    W = W_unmask * chord_mask(N)

    # WV
    V0 = np.matmul(W, V)

    return V0

# ____________________________________________________
# END OF NEW


def backward(parameters, N, V, memory):
    """
    #calculates gradients of J respect to V and F
    :return:
    """
    grad_values = {}
    L = int(math.log(N, 2))
    K = 2
    grad_values['dv'] = 2 * np.matmul(1 / memory['W' + str(L)].T, parameters['V'] - V)
    for i in range(L):
        df = np.zeros((N, 2))
        for j in range(N):
            for k in range(K):
                tmp1 = 2 * np.matmul(1 / memory['W' + str(L - i)].T, parameters['V' + str(L - i)] - V)
                tmp2 = np.matrix(parameters['V' + str(L - i)][(j + np.power(2, k)) % N]).T
                tmp = np.matmul(tmp1, tmp2)
                df[j][k] = tmp[j][0]
        grad_values['dF' + str(L - i)] = df

    return grad_values


def update_parameters(parameters, grad_values, learning_rate, N):
    """
    :param parameters: from previous epoch
    :param grad_values: from backward
    :param learning_rate:
    :param N:
    :return:
    """
    L = int(math.log(N, 2))
    parameters['V'] -= learning_rate * grad_values["dv"]
    for i in range(L):
        parameters['F' + str(L - i)] -= learning_rate * grad_values["dF" + str(L - i)]
        parameters['V' + str(L - i)] -= learning_rate * grad_values["dv"]

    return parameters


def traning(epochs, learning_rate, N):
    """
    normal training process as in NN
    :param epochs:
    :param learning_rate:
    :param N:
    :return:
    """
    parameters = init_layers(N)
    cost_history = []
    for i in range(epochs):
        V, memory = forward(parameters)
        cost = get_cost_value(V, parameters['V_copy'])
        grad_values = backward(parameters, N, V, memory)
        parameters = update_parameters(parameters, grad_values, learning_rate, N)
        cost_history.append(cost)
        print(cost)
        # print(V)

    return parameters, cost_history


if __name__ == "__main__":
    traning(2, 0.01, 16)
