"""
An implementation of a small 1-layer model.
Testing environment

"""

# Imports
import sys
import math
import numpy as np
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


# Activation functions and its derivatives
def relu(x):
    return max(0, x)


def leaky_relu(x, slope=0.01):
    return max(slope * x, x)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_relu(x):
    return 1 if x > 0 else 0


def d_leaky_relu(x, slope=0.01):
    return 1 if x > 0 else slope


def d_tanh(x):
    return 1 - tanh(x) ** 2


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


activation_g = relu
activation_f = relu
d_activation_g = d_relu
d_activation_f = d_relu


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
    assert transform in transform_to_func.keys(
    ), f"Unknown transformation {transform}," f" available: {transform_to_func.keys()}"
    return transform_to_func[transform](X)


def init_f():
    """
    Initializes f (d -> N)
    :param N: Sequence length
    :param d: Embedding size
    :return: None
    """
    weights = np.zeros((N, d))
    bias = np.zeros(N)
    grad_weights = np.zeros((N * d, N, N))
    grad_bias = np.zeros(N, N)

    parameters['f']['weights'] = weights
    parameters['f']['bias'] = bias
    parameters['f']['grad_weights'] = grad_weights
    parameters['f']['grad_bias'] = grad_bias


def init_g():
    """
    Initializes g (d -> d)
    :param N: Sequence length
    :param d: Embedding size
    :return: None
    """
    weights = np.zeros((d, d))
    bias = np.zeros(d)
    grad_weights = np.zeros((d * d, N, d))
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
    Forward pass X -> V0. Also computes d_activation for
    :param parameters: Current parameters of f and g
    :return: V0, V, W, d_activation values for the backward pass
    """

    # V
    V = np.zeros(N, d)
    d_activation_values_g = np.zeros(N, d)
    assert np.shape(V) == np.shape(X), "V and X have different size"
    xi = parameters['g']['weights']
    bias = parameters['g']['bias']

    # rows
    for i in range(N):
        # columns
        for j in range(d):
            # bias[j] + X[i, 0] * xi[j, 0] + X[i, 1] * xi[j, 1]
            value = bias[j] + np.sum([X[i, _] * xi[j, _] for _ in range(d)])
            V[i, j] = activation_g(value)
            d_activation_values_g[i, j] = d_activation_g(value)

    # W
    W_unmask = np.zeros(N, N)
    d_activation_values_f = np.zeros(N, N)
    theta = parameters['f']['weights']
    bias = parameters['f']['bias']

    # rows
    for i in range(N):
        # columns
        for j in range(N):
            # bias[j] + X[i, 0] * theta[j, 0] + X[i, 1] * theta[j, 1] + X[i, 2] * theta[j, 2]
            value = bias[j] + np.sum([X[i, _] * theta[j, _] for _ in range(N)])
            W_unmask[i, j] = activation_f(value)
            d_activation_values_f[i, j] = d_activation_f(value)

    # masking W (elementwise product)
    W = W_unmask * chord_mask(N)
    d_activation_values_f = d_activation_values_f * chord_mask(N)

    # WV (matrix multiplication)
    V0 = np.matmul(W, V)

    return V0, V, W, d_activation_values_g, d_activation_values_f


def backward(parameters, V0, V, W, d_activation_values_g, d_activation_values_f, V_gt):
    """
    Calculates gradients of the Loss function (J) w.r.t (theta, bias) and (xi, bias)
    and stores it in parameters['grad_weights'] and parameters['grad_bias']
    for f and g respectively.

    :param parameters: parameters of the model, gradient containers
    :param V0: final result of the forward pass
    :param V: tensor of size (Nxd)
    :param W: tensor of size (NxN), after masking
    :param d_activation_values_g: tensor of size (Nxd)
    :param d_activation_values_f: tensor of size (NxN)
    :param V_gt: tensor of size (Nxd), ground truth
    :return: None
    """

    # TODO: add biases

    dj_dv0 = -2 * (V_gt - V0).T               # (d, N)
    dv0_dw = V                                # (N, d)
    dv0_dv = W                                # (N, N)
    dj_dw = dj_dv0 * dv0_dw                   # (d, d)
    dj_dv = dj_dv0 * dv0_dv                   # (d, N)

    # g


    # f


# ____________________________________________________
# END OF NEW


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
        parameters['F' + str(L - i)] -= learning_rate * \
            grad_values["dF" + str(L - i)]
        parameters['V' + str(L - i)] -= learning_rate * grad_values["dv"]

    return parameters


def training(epochs, learning_rate, N):
    """
    normal training process as in NN
    :param epochs:
    :param learning_rate:
    :param N:
    :return:
    """
    cost_history = []
    for i in range(epochs):
        V, memory = forward(parameters)
        cost = calculate_loss(V, parameters['V_copy'])
        grad_values = backward(parameters, N, V, memory)
        parameters = update_parameters(
            parameters, grad_values, learning_rate, N)
        cost_history.append(cost)
        print(cost)
        # print(V)

    return parameters, cost_history


if __name__ == "__main__":
    training(2, 0.01, 16)
