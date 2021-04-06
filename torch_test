from SMF_torch import *
from permute_data import *
import torch

def training(X, X_gt):
    """
    Test on a small case
    :return:
    """
    N, D_in, H1, H2, n_layer = 16, 16, 16, 16, 4
    #X = torch.randn(N, D_in)
    #X_gt = torch.randn(N, D_in)
    model = SMFNet(D_in, H1, H2, n_layer)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(100):
        V0 = model(X, n_layer)
        loss = criterion(X_gt.float(), V0)
        loss.backward()
        optimizer.step()
        print(loss)

    return V0


if __name__ == '__main__':
    X, X_gt = generate_permute_data_sine(16, 16, noise=0.8)
    X = torch.tensor(X)
    X_gt = torch.tensor(X_gt)
    V0 = training(X, X_gt)
    V0 = V0.detach().numpy()
    import matplotlib.pyplot as plt
    i = 10
    plt.plot(X_gt[i, :], 'r')
    plt.plot(V0[i, :], 'b')
    plt.show()
