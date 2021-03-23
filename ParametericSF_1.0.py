import numpy as np
import math


def init_layers(N):
    """
    initialize F, V and W for each layer
    especially, parameters['V'] is the origin V generated by random instead of g(),
    :param N:
    :return: parameters of each layer
    """
    parameters = {}
    parameters['V'] = np.random.rand(N, 2)
    parameters['V_copy'] = parameters['V']
    L = int(math.log(N, 2))
    for i in range(L):
        parameters['F' + str(L-i)] = np.random.rand(N, 2)
        parameters['W' + str(L-i)] = np.random.rand(N, N)
        parameters['V' + str(L-i)] = np.random.rand(N, 2)

    return parameters


def get_cost_value(V_init, V):
    """
    MSE is the cost function to calculate the distance from V_init to V0
    :param V_init: the origin V from g()
    :param V: the final result of the forward process
    :return:
    """
    cost = np.square(np.subtract(V_init, V)).mean()
    return cost


def forward(parameters):
    """
    forward process of the matrix multiplication
    :return: V = V.T *W
    """
    N = np.shape(parameters['V'])[0]
    K = 2
    L = int(math.log(N, 2))
    V = parameters['V']
    memory = {}
    for l in range(L):
        F = parameters['F' + str(L-l)]
        for i in range(N):
            for k in range(K):
                parameters['W'+str(L-l)][i][(i + np.power(2, k)) % N] = F[i][k]
        #print(parameters['W'+str(L-l)])
        memory['W'+str(L-l)] = parameters['W'+str(L-l)]
        memory['V'+str(L-l)] = parameters['V'+str(L-l)]
        V = np.matmul(parameters['W'+str(L-l)].T, V)

    return V, memory


def backward(parameters, N, V, memory):
    """
    #calculates gradients of J respect to V and F
    :return:
    """
    grad_values = {}
    L = int(math.log(N, 2))
    K = 2
    grad_values['dv'] = 2 * np.matmul(1/memory['W'+str(L)].T, parameters['V']-V)
    for i in range(L):
        df = np.zeros((N, 2))
        for j in range(N):
            for k in range(K):
                tmp1 = 2* np.matmul(1/memory['W'+str(L-i)].T, parameters['V'+str(L-i)]-V)
                tmp2 = np.matrix(parameters['V'+str(L-i)][(j + np.power(2, k)) % N]).T
                tmp = np.matmul(tmp1, tmp2)
                df[j][k] = tmp[j][0]
        grad_values['dF'+str(L-i)] = df

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
        parameters['F'+str(L-i)] -= learning_rate * grad_values["dF" + str(L-i)]
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
        #print(V)

    return parameters, cost_history


if __name__ == "__main__":
    traning(2, 0.01, 16)