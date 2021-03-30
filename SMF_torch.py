import torch
import numpy as np


class SMFNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2):
        """
        :param D_in: Embedded size of original data
        :param H1: Hidden layer dimension of g
        :param H2: Hidden layer dimension of f
        """
        super(SMFNet, self).__init__()
        self.f_linear = torch.nn.Linear(D_in, H2, bias=False)
        self.g_linear = torch.nn.Linear(D_in, H1, bias=False)
        theta = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
        theta = torch.tensor(theta)
        xi = [[0.05, 0.02], [0.01, 0.03]]
        xi = torch.tensor(xi)
        f_bias = [0.01, 0.02, 0.03]
        g_bias = [0.03, 0.02]
        f_bias = torch.tensor(f_bias)
        g_bias = torch.tensor(g_bias)
        self.f_linear.weight = torch.nn.Parameter(theta)
        self.f_linear.bias = torch.nn.Parameter(f_bias)
        self.g_linear.weight = torch.nn.Parameter(xi)
        self.g_linear.bias = torch.nn.Parameter(g_bias)
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

    def forward(self, X):
        """
        :param X: Nxd data
        :return: V(M-1) M is the layer number.
        """
        N = X.shape[0]
        V = torch.zeros(N, 2)
        W = torch.zeros(N, N)
        chord_mask = self.make_chord(N)
        for i in range(N):
            v_out = self.relu(self.g_linear(X[i]))
            f_out = self.relu(self.f_linear(X[i]))
            V[i] = v_out
            W[i] = f_out

        print(V)
        W = torch.mul(W, chord_mask)
        print(W)
        V0 = torch.matmul(W, V)

        return V0


def training():
    """
    Test on a small case
    :return:
    """
    N, D_in, H1, H2 = 3, 2, 2, 3
    #X = torch.randn(N, D_in)
    #X_gt = torch.randn(N, D_in)
    X = [[2.0000, -1.0000], [-3.0000, -4.0000], [-2.0000, 2.0000]]
    X_gt = [[1.0000, 4.0000], [5.0000, -6.0000], [3.0000, 2.0000]]
    X = torch.tensor(X)
    X_gt = torch.tensor(X_gt)
    model = SMFNet(D_in, H1, H2)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(1):
        V0 = model(X)
        print(V0)
        loss = criterion(X_gt, V0)
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("weight")
        print(model.f_linear.weight.grad)
        print(model.f_linear.bias.grad)
        print(model.g_linear.weight.grad)
        print(model.g_linear.bias.grad)


if __name__ == '__main__':
    training()


