import torch
import numpy as np


class SMFNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, n_layer):
        """
        :param D_in: Embedded size of original data
        :param H1: Hidden layer dimension of g
        :param H2: Hidden layer dimension of f
        :param n_layer: Layer number of SMFNet
        :param depth: layer number of f and g
        """
        super(SMFNet, self).__init__()
        self.n_layer = n_layer
        self.f_array = []
        self.g_linears = torch.nn.Sequential(
            torch.nn.Linear(D_in, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, H1),
            torch.nn.ReLU(),
        )
        for m in range(n_layer):
            #f_linears = torch.nn.ModuleList(torch.nn.Linear(D_in, D_in) for d in range(depth-1))
            #f_linears.append(torch.nn.Linear(D_in, H2))
            f_linears = torch.nn.Sequential(
            torch.nn.Linear(D_in, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, H2),
            torch.nn.ReLU(),
            )
            self.f_array.append(f_linears)

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

    def forward(self, X):
        """
        :param X: Nxd data
        :return: V(M-1) M is the layer number.
        """
        N = X.shape[1]
        W = [torch.zeros(N, N)]*self.n_layer
        chord_mask = self.make_chord(N)
        #print(V0)
        V0 = self.g_linears(X.float())
        for m in range(self.n_layer):
            F = self.f_array[m](X.float())
            W[m] = F*chord_mask
            V0 = torch.matmul(W[m], V0)

        return V0
