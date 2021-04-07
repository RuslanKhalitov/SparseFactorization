import torch
import numpy as np


class SMFNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, n_layer, depth):
        """
        :param D_in: Embedded size of original data
        :param H1: Hidden layer dimension of g
        :param H2: Hidden layer dimension of f
        :param n_layer: Layer number of SMFNet
        :param depth: layer number of f and g
        """
        super(SMFNet, self).__init__()
        self.f_array = []
        self.g_linears = torch.nn.ModuleList(torch.nn.Linear(D_in, H1) for d in range(depth))
        for m in range(n_layer):
            f_linears = torch.nn.ModuleList(torch.nn.Linear(D_in, D_in) for d in range(depth - 1))
            f_linears.append(torch.nn.Linear(D_in, H2))
            self.f_array.append(f_linears)
        self.relu = torch.nn.ReLU()
        self.n_layer = n_layer
        self.depth = depth

    @staticmethod
    def make_chord(N):
        """
        :param N: Length of original data
        :return:
        """
        chord_mask = torch.eye(N)
        for i in range(N):
            for k in range(2):
                chord_mask[i][(i + np.power(2, k) - 1) % N] = 1

        return chord_mask

    def forward(self, X):
        """
        :param X: Nxd data
        :return: V(M-1) M is the layer number.
        """
        N = X.shape[0]
        W = [torch.zeros(N, N)] * self.n_layer
        chord_mask = self.make_chord(N)
        V0 = X.float()
        for d in range(self.depth):
            V0 = self.relu(self.g_linears[d](V0))
        for m in range(self.n_layer):
            F = X.float()
            for d in range(self.depth):
                F = self.relu(self.f_array[m][d](F))
            W[m] = F * chord_mask
            V0 = torch.matmul(W[m], V0)

        return V0
