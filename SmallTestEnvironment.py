"""
An implementation of a multilayer module.
Testing environment

"""

# Imports
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
    return np.maximum(0, x)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_relu(dA, z):
    d = np.array(dA, copy=True)
    d[z < 0] = 0.
    return d


def d_tanh(dA, z):
    return dA * (1 - tanh(out) ** 2)


def d_sigmoid(dA, z):
    sig = sigmoid(z)
    return dA * sig * (1 - sig)


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


def single_layer_forward_g(A_prev, parameters):
    """
    Emdedding from X to V line by line
    :param A_prev: activation result from previous layer
    :param parameters:
    :return: V with the same dimesntion with X
    """
    # V
    V = np.zeros(N, d)
    values_g = np.zeros(N, d)
    assert np.shape(V) == np.shape(A_prev), "V and X have different size"
    xi = parameters['g']['weights']
    bias = parameters['g']['bias']

    # rows
    for i in range(N):
        # columns
        for j in range(d):
            # bias[j] + X[i, 0] * xi[j, 0] + X[i, 1] * xi[j, 1]
            value = bias[j] + np.sum([A_prev[i, _] * xi[j, _] for _ in range(d)])
            V[i, j] = activation_g(value)
            values_g[i, j] = value

    return V, values_g


def single_layer_forward_f(A_prev, parameters):
    """
    Generate W
    :param A_prev: activation result from previous layer
    :param parameters:
    :return: masked W (N X N)
    """

    W_unmask = np.zeros(N, N)
    values_f = np.zeros(N, N)
    theta = parameters['f']['weights']
    bias = parameters['f']['bias']

    for i in range(N):
         for j in range(N):
            # bias[j] + X[i, 0] * theta[j, 0] + X[i, 1] * theta[j, 1] + X[i, 2] * theta[j, 2]
            value = bias[j] + np.sum([A_prev[i, _] * theta[j, _] for _ in range(N)])
            W_unmask[i, j] = activation_f(value)
            values_f[i, j] = value

    # masking W (elementwise product)
    W = W_unmask * chord_mask(N)
    values_f = values_f * chord_mask(N)

    return W, values_f


def full_forward_g(X, n_layers, parameters):
    """
    The whole neural network g.
    :param X: original data
    :param n_layers: layers of g
    :param parameters:
    :return: The final V after N layers and the memory storing Z and A.
    """
    memory_g = {}
    A_curr = X

    for idx, layer in enumerate(n_layers):
        layer_idx = idx + 1
        A_prev = A_curr

        A_curr, Z_curr = single_layer_forward_g(A_prev, parameters)

        memory_g["A_g" + str(idx)] = A_prev
        memory_g["Z_g" + str(layer_idx)] = Z_curr

    return A_curr, memory_g


def full_forward_f(X, n_layers, parameters):
    """
    The core of the whole framework. Forward progress of neural network f.
    :param X: original data
    :param n_layers:
    :param parameters:
    :return: The final W and the memory storing Z and A.
    """
    memory_f = {}
    A_curr = X

    for idx, layer in enumerate(n_layers):
        layer_idx = idx + 1
        A_prev = A_curr

        A_curr, Z_curr = single_layer_forward_f(A_prev, parameters)

        memory_f["A_f" + str(idx)] = A_prev
        memory_f["Z_f" + str(layer_idx)] = Z_curr

    return A_curr, memory_f


# Backward part
# _______________________________________________


def single_layer_backward_g(dA_curr, parameters, Z_curr, A_prev):
    """
    calculate the derivative of A_curr with respect to prev_A, W_curr and b_curr
    :param dA_curr: from the following layer of g
    :param parameters:
    :param Z_curr: output of current layer
    :param A_prev: activation result of previous layer.
    :return:
    """
    # A_prev is V
    m = A_prev.shape[1]
    xi = parameters['g']['weights']
    bias = parameters['g']['bias']

    dZ_curr = d_relu(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(xi.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def single_layer_backward_f(dA_curr, parameters, Z_curr, A_prev):
    """
    calculate the derivative of A_curr with respect to prev_A, W_curr and b_curr
    :param dA_curr: from the following layer of f
    :param parameters:
    :param Z_curr: output of current layer
    :param A_prev: activation result of previous layer.
    :return:
    """
    # A_prev is W
    m = A_prev.shape[1]
    theta = parameters['f']['weights']
    bias = parameters['f']['bias']

    dZ_curr = d_relu(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(theta.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(V_gt, V, W, memory_f, memory_g, parameters, n_layers):
    """
    The core of the whole framework. backward progress of neural network f.
    :param V_gt: Ground truth
    :param V: The final V from g
    :param W: The final W from f
    :return:
    """

    # necessary derivatives in matrix form
    V0 = np.dot(W, V)                         # (d, N)
    dj_dv0 = -2 * (V_gt - V0).T               # (d, N)
    dv0_dw = V                                # (N, d)
    dv0_dv = W                                # (N, N)
    dj_dw = dj_dv0 * dv0_dw                   # (d, d)
    dj_dv = dj_dv0 * dv0_dv                   # (d, N)

    # V
    for layer_idx_prev, layer in reversed(list(enumerate(n_layers))):
        layer_idx_curr = layer_idx_prev + 1
        dA_curr = dj_dv

        A_prev = memory_g["A_g" + str(layer_idx_prev)]
        Z_curr = memory_g["Z_g" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_g(
            dA_curr, parameters, Z_curr, A_prev
        )

        parameters['g']['grad_weights' + str(layer_idx_curr)] = dW_curr
        parameters['g']['grad_bias' + str(layer_idx_curr)] = db_curr

    # W
    for layer_idx_prev, layer in reversed(list(enumerate(n_layers))):
        layer_idx_curr = layer_idx_prev + 1
        dA_curr = dj_dw

        A_prev = memory_f["A_f" + str(layer_idx_prev)]
        Z_curr = memory_f["Z_f" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_f(
            dA_curr, parameters, Z_curr, A_prev
        )

        parameters['f']['grad_weights' + str(layer_idx_curr)] = dW_curr
        parameters['f']['grad_bias' + str(layer_idx_curr)] = db_curr

    return parameters


def update(parameters, n_layers):
    """
    updates parameters in g and f according to the update rule.
    :param parameters:
    :param n_layers:
    :return:
    """
    #V
    for layer_idx, layer in enumerate(n_layers):
        parameters['g']["W" + str(layer_idx)] -= LRg * parameters["grad_weights" + str(layer_idx)]
        parameters['g']["b" + str(layer_idx)] -= LRg * parameters["grad_bias" + str(layer_idx)]

    #W
    for layer_idx, layer in enumerate(n_layers):
        parameters['f']["W" + str(layer_idx)] -= LRf * parameters["grad_weights" + str(layer_idx)]
        parameters['f']["b" + str(layer_idx)] -= LRf * parameters["grad_bias" + str(layer_idx)]

    return parameters


def train(X, V_gt, n_layers, epochs):
    """
    Train a model.
    :param X: original data
    :param V_gt: ground truth
    :param n_layers:
    :param epochs:
    :return:
    """
    init_f()
    init_g()
    cost_history = []

    for i in range(epochs):
        W, memory_f = full_forward_f(X, n_layers, parameters)
        V, memory_g = full_forward_g(X, n_layers, parameters)
        cost = calculate_loss(V_gt, V0)
        cost_history.append(cost)

        grads_values = full_backward_propagation(V_gt, V, W, memory_f, memory_g, parameters, n_layers)
        params_values = update(parameters, n_layers)

    return parameters, cost_history


if __name__ == '__main__':
    """
    test on a 3X2 matrix
    """
    X = get_X(3, 2)
    V_gt = get_Vgt(X)
    train(X, V_gt, 2, 5)
