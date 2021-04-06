import torch
import numpy as np


class SMFNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, n_layer):
        """
        :param D_in: Embedded size of original data
        :param H1: Hidden layer dimension of g
        :param H2: Hidden layer dimension of f
        """
        super(SMFNet, self).__init__()
        self.f_array = []
        for m in range(n_layer):
            self.f_array.append(torch.nn.Linear(D_in, H2))
        self.g_linear = torch.nn.Linear(D_in, H1)
        self.relu = torch.nn.ReLU()

    def make_chord(self, N):
        """
        :param N: Length of original data
        :return:
        """
        chord_mask = torch.eye(N)
        for i in range(N):
            for k in range(2):
                chord_mask[i][(i + np.power(2, k) - 1) % N] = 1

        return chord_mask

    def forward(self, X, n_layer):
        """
        :param X: Nxd data
        :return: V(M-1) M is the layer number.
        """
        N = X.shape[0]
        W = [torch.zeros(N, N)]*n_layer
        chord_mask = self.make_chord(N)
        V0 = self.relu(self.g_linear(X.float()))
        #print(V0)
        for m in range(n_layer):
            F = self.relu(self.f_array[m](X.float()))
            W[m] = F*chord_mask
            V0 = torch.matmul(W[m], V0)

        return V0



