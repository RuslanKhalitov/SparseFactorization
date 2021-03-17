import torch
import numpy as np


class SMFNet(torch.nn.Module):
    def __init__(self, D_in, H):
        super(SMFNet, self).__init__()
        self.f_linear = torch.nn.Linear(D_in, H)
        self.g_linear = torch.nn.Linear(D_in, H)

    def forward(self, X):
        N = X.shape[0]
        V = torch.zeros(N, 2)
        F = torch.zeros(N, 2)
        W = torch.zeros(N, N)
        for i in range(N):
            v_out = self.g_linear(X[i])
            f_out = self.g_linear(X[i])
            V[i] = v_out
            F[i] = f_out
        for i in range(N):
            for k in range(2):
                W[i][(i + np.power(2, k)) % N] = F[i][k]
        V0 = torch.matmul(W, V)

        return V0


def training():
    N, D_in, H = 3, 2, 2

    X = torch.randn(N, D_in)
    X_gt = torch.randn(N, D_in)
    #print(X_gt)
    model = SMFNet(D_in, H)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for i in range(1000):
        V0 = model(X)
        loss = criterion(X_gt, V0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        #print(V0)


if __name__ == '__main__':
    training()


