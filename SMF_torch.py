import torch
import numpy as np


class SMFNet(torch.nn.Module):
    def __init__(self, D_in, H):
        super(SMFNet, self).__init__()
        self.f_linear = torch.nn.Linear(D_in, H, bias=False)
        self.g_linear = torch.nn.Linear(D_in, H, bias=False)
        self.f_linear.weight.data.fill_(0.01)
        self.g_linear.weight.data.fill_(0.05)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        N = X.shape[0]
        V = torch.zeros(N, 2)
        F = torch.zeros(N, 2)
        W = torch.zeros(N, N)
        for i in range(N):
            v_out = self.relu(self.g_linear(X[i]))
            f_out = self.relu(self.f_linear(X[i]))
            V[i] = v_out
            F[i] = f_out
        for i in range(N):
            for k in range(2):
                W[i][(i + np.power(2, k) - 1) % N] = F[i][k]
        V0 = torch.matmul(W, V)

        return V0


def training():
    N, D_in, H = 3, 2, 2
    #X = torch.randn(N, D_in)
    #X_gt = torch.randn(N, D_in)
    X = [[1.0000, 2.0000], [3.0000, 4.0000], [2.0000, 0.0000]]
    X_gt = [[2.0000, 4.0000], [6.0000, 8.0000], [4.0000, 0.0000]]
    X = torch.tensor(X)
    X_gt = torch.tensor(X_gt)
    print(X)
    print(X_gt)
    model = SMFNet(D_in, H)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for i in range(1):
        V0 = model(X)
        print(V0)
        loss = criterion(X_gt, V0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        #print(V0)


if __name__ == '__main__':
    training()


