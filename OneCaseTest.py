from SmallTestEnvironment import *
import numpy as np

if __name__ == '__main__':
    parameters['g']['weights'] = np.array([[0.03, -0.03], [-0.2, 0.03]])
    parameters['g']['bias'] = np.array([0.03, 0.02])

    parameters['f']['weights'] = np.array([[0.1, 0.02], [-0.03, 0.04], [0.05, -0.6]])
    parameters['f']['bias'] = np.array([0.2, -0.02, 0.03])

    X = np.array([[1.0000, -3.0000], [-30.0000, 4.0000], [-2.0000, 7.0000]])
    X_gt = np.array([[2.0000, 6.0000], [60.0000, 8.0000], [4.0000, 14.0000]])

    W, mem_W = full_forward_f(X, 1, parameters)
    print('W \n', W)
    V, mem_V = full_forward_g(X, 1, parameters)
    print('V \n', V)
    V0 = np.dot(W, V)
    print('V0 \n', V0)

    # backward
    full_backward_propagation(X_gt, V, W, mem_W, mem_V, parameters, 1)
    print('Grad values of theta \n', parameters['f']['grad_weights1'])
    print('Grad values of beta_f \n', parameters['f']['grad_bias1'])
    print('Grad values of xi \n', parameters['g']['grad_weights1'])
    print('Grad values of beta_g \n', parameters['g']['grad_bias1'])