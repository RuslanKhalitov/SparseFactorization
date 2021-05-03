import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os

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
    def __init__(self, n_class, n_W, n_vec, n_dim, n_hidden_f=10, n_hidden_g=10):
        super(InteractionModule, self).__init__()
        self.n_vec = n_vec
        self.n_dim = n_dim
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        self.g = VIdenticalModule()
        self.final = nn.Linear(self.n_vec*self.n_dim, n_class, bias=True)
#         self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        V = self.g(data)
        for f in self.fs[::-1]:
            W = f(data)
            V = W @ V
        V = self.final(V.view(data.size(0), -1))
        return V


def generate_3class_data(n_data, n_vec=4, n_dim=3):
    assert n_data % 3 == 0, 'Please use the n_data number multiple by 3'
    n_each = int(n_data / 3)

    # first dataset x0 = (-1) * x1 + 2 * x2
    first_data = []
    first_labels = []
    for i in range(n_each):
        X = torch.randn(n_vec, n_dim)
        X[0] = (0.5) * X[1] + (0.5) * X[2]
        first_data.append(X)
        first_labels.append(0)

    # second dataset x1 = 0.5 * x0 + (-2) * x3
    second_data = []
    second_labels = []
    for i in range(n_each):
        X = torch.randn(n_vec, n_dim)
        X[0] = (0.3) * X[1] + (0.7) * X[2]
        second_data.append(X)
        second_labels.append(1)

        # third dataset x3 = 4 * x1 + (-4) * x2
    third_data = []
    third_labels = []
    for i in range(n_each):
        X = torch.randn(n_vec, n_dim)
        X[0] = (0.9) * X[1] + (0.1) * X[2]
        third_data.append(X)
        third_labels.append(2)

    all_data = first_data + second_data + third_data
    all_labels = first_labels + second_labels + third_labels
    return all_data, all_labels


class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, mode, data, labels):
        print(f'Creating data loader - {mode}')
        assert mode in ['train', 'test']
        self.data = data
        self.labels = labels
        assert len(self.data) == len(self.labels),\
            "The number of samples doesn't match the number of labels"

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = data[index]
        Y = labels[index]
        return (X, Y)

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)

    n_data = 12000
    n_class = 3
    n_W = 2
    n_vec = 4
    n_dim = 3
    n_hidden_f = 20
    n_hidden_g = 3
    batch_size = 120
    data, labels = generate_3class_data(n_data=n_data, n_vec=n_vec, n_dim=n_dim)
    trainset = DatasetCreator(
        mode='train',
        data=data,
        labels=labels
    )
    data_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True
    )
    net = InteractionModule(n_class, n_W, n_vec, n_dim, n_hidden_f, n_hidden_g)
    net.apply(weights_init)
    print(net)
    loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Training
    n_epochs = 5000
    test_freq = 10
    losses = []
    for epoch in range(n_epochs):
        for i, (X, Y) in enumerate(data_loader):
            optimizer.zero_grad()
            pred = net(X)
            output = loss(pred, Y)
            output.backward()
            optimizer.step()

        if epoch % test_freq == 0:
            net.eval()
            with torch.no_grad():
                total_loss = 0.0
                for i, (X, Y) in enumerate(data_loader):
                    pred = net(X)
                    total_loss += loss(pred, Y)
            mean_loss = total_loss / n_data
            print("epoch=%d/%d, mean_loss=%.10f" % (epoch, n_epochs, mean_loss))
            losses.append(float(mean_loss))
            net.train()
            if mean_loss < 0.0002:
                optimizer = optim.Adam(net.parameters(), lr=1e-5)

    # Plot
    plt.plot([test_freq * i for i in range(len(losses))], losses)
    plt.xlabel('n_epochs')
    plt.ylabel('mean_loss')
    plt.yscale('log')
    plt.ylim(10 ** -4.3, 10 ** -2)


    # Test

    with torch.no_grad():
        X = data[0].unsqueeze_(0)
        Y = labels[0]
        print('label \n', Y)
        print('predicted \n', net(X))

        X = data[5000].unsqueeze_(0)
        Y = labels[5000]
        print('label \n', Y)
        print('predicted \n', net(X))

        X = data[10000].unsqueeze_(0)
        Y = labels[10000]
        print('label \n', Y)
        print('predicted \n', net(X))
