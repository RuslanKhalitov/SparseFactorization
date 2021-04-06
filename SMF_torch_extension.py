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
        V0 = self.relu(self.g_linear(X))
        for m in range(n_layer):
            F = self.relu(self.f_array[m](X))
            W[m] = F*chord_mask
            V0 = torch.matmul(W[m], V0)

        return V0


def training():
    """
    Test on a small case
    :return:
    """
    N, D_in, H1, H2, n_layer = 16, 2, 2, 16, 4
    X = torch.randn(N, D_in)
    X_gt = torch.randn(N, D_in)
    #X = [[2.0000, -1.0000], [-3.0000, -4.0000], [-2.0000, 2.0000]]
    #X_gt = [[1.0000, 4.0000], [5.0000, -6.0000], [3.0000, 2.0000]]
    #X = torch.tensor(X)
    #X_gt = torch.tensor(X_gt)
    model = SMFNet(D_in, H1, H2, n_layer)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(50):
        V0 = model(X, n_layer)
        #print(V0)
        loss = criterion(X_gt, V0)
        loss.backward()
        optimizer.step()
        print(loss)
        #print("weight")
        #print(model.f_array[0].weight.grad)
        #print(model.f_array[0].bias.grad)
        #print(model.g_linear.weight.grad)
        #print(model.g_linear.bias.grad)


if __name__ == '__main__':
    training()


