import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(module.weight, 0.0, 1e-2)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.normal_(module.bias, 0.0, 1e-2)


class WModule(nn.Module):
    def __init__(self, n_vec, n_dim, n_hidden):
        super(WModule, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_vec)
        )

    def forward(self, data):
        return self.network(data)


class VIdenticalModule(nn.Module):
    def __init__(self):
        super(VIdenticalModule, self).__init__()

    def forward(self, data):
        return data


class InteractionModule(nn.Module):
    def __init__(self, n_W, n_vec, n_dim, n_hidden_f=10, n_hidden_g=10):
        super(InteractionModule, self).__init__()
        self.fs = nn.ModuleList([WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)])
        self.g = VIdenticalModule()

    def forward(self, data):
        V = self.g(data)
        for f in self.fs[::-1]:
            W = f(data)
            V = W @ V
        return V


def generate_data(n_data, n_vec, n_dim):
    A = torch.rand(n_vec, n_vec) - 0.5
#     vec = torch.rand(n_vec, n_dim)
    all_data = [torch.randn(n_vec, n_dim) for _ in range(n_data)]
    all_data_gt = [A @ all_data[i] for i in range(n_data)]
#     return all_data, all_data_gt, A, vec
    return all_data, all_data_gt, A


def calculate_mean_loss(net, loss, all_data, all_data_gt):
    n_data = len(all_data)
    with torch.no_grad():
        total_loss = 0.0
        for i in range(n_data):
            data = all_data[i]
            data_gt = all_data_gt[i]
            data_pred = net(data)
            total_loss += loss(data_pred, data_gt)
        return total_loss/len(all_data)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    n_data = 10000
    n_W = 4
    n_vec = 3
    n_dim = 4
    n_hidden_f = 50
    n_hidden_g = 3
    net = InteractionModule(n_W, n_vec, n_dim, n_hidden_f, n_hidden_g)
    net.apply(weights_init)
    loss = nn.MSELoss()
    all_data, all_data_gt, A = generate_data(n_data, n_vec, n_dim)

    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    max_epoch = 300000
    for epoch in range(max_epoch):
        i = random.randint(0, n_data - 1)
        data = all_data[i]
        data_gt = all_data_gt[i]
        data_pred = net(data)
        output = loss(data_pred, data_gt)
        output.backward()
        optimizer.step()

        if epoch % (max_epoch / 10) == 0:
            mean_loss = calculate_mean_loss(net, loss, all_data, all_data_gt)
            print("epoch=%d/%d, mean_loss=%.10f" % (epoch, max_epoch, mean_loss))


