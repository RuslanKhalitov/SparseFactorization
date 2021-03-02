import numpy as np
import math


def fm(N):
    """
    be supposed to be a neurual network
    :param N: dimension 0 of the data
    :return: N*d dimension random binary matrix
    """
    F = np.random.randint(2, size=(N, 2))
    return F


def g(N):
    """
    :param N: dimension 0 of the data
    :return: N*d dimension random binary matrix
    """
    V = np.random.randint(2, size=(N, 2))
    return V


def forward(V):
    """
    forward process of the matrix multiplication
    :param V:
    :return: V = V.T *W
    """
    W = np.zeros((16,16))
    N = np.shape(W)[0]
    K = 2
    L = int(math.log(N, 2))
    w = []
    v = []
    for l in range(L):
        F = fm(16)
        for i in range(N):
            for k in range(K):
                W[i][(i + np.power(2, k)) % N] = F[i][k]
        #print(W)
        V = np.matmul(W, V)
        w.append(W)
        v.append(V)
    return V, w, v


def calculateDelta():
    """
    :return:
    """
    return


def backward(w, v):
    """
    #1. multiply output delta and input activation to get the gradient of the weight
    :return:
    """
    F = []
    for i in range(len(w)):
        delta = calculateDelta()
        v[i] = w[i].T.dot(delta)
        F[i] = delta.dot(v[i+pow(2, k):].T)

    return v, F


def main():
    """
    set 16 as a default dimension to test
    """
    V = g(16)
    V, w, v = forward(V)
    print(len(w))
    print(len(v))


if __name__ == "__main__":
    main()