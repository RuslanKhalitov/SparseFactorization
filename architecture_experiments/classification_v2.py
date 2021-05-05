import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from ChordMatrix import chord_mask
import os
import math


def weights_init(module):
    if type(module) == torch.nn.Linear:
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


class VModule(nn.Module):
    def __init__(self, n_dim, n_hidden):
        super(VModule, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_dim)
        )

    def forward(self, data):
        return self.network(data)


class VIdenticalModule(nn.Module):
    def __init__(self):
        super(VIdenticalModule, self).__init__()

    def forward(self, data):
        return data


class InteractionModule(nn.Module):
    def __init__(self, n_class, n_W, n_vec, n_dim, n_hidden_f=10, n_hidden_g=10, masking=False, with_g = False):
        super(InteractionModule, self).__init__()
        self.n_vec = n_vec
        self.n_dim = n_dim
        self.chord_mask = chord_mask(self.n_vec)
        self.masking = masking
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        if not with_g:
            self.g = VIdenticalModule()
        else:
            self.g = VModule(n_dim, n_hidden_g)
            #print(self.g)
        self.final = nn.Linear(self.n_vec*self.n_dim, n_class, bias=True)
#         self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        V = self.g(data)
        for f in self.fs[::-1]:
            if self.masking:
                W = f(data) * self.chord_mask
            else:
                W = f(data)
            V = W @ V
        V = self.final(V.view(data.size(0), -1))
        return V


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
        X = self.data[index]
        Y = self.labels[index]
        return (X, Y)

    def __len__(self):
        return len(self.labels)


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


def plot_metrics(losses, N, d, n_W, masking, with_g):
    plt.gca().cla()
    figure_name = f'{n_W}layers{N}N_{d}d_mask{masking}with_g{with_g}'
    plt.plot([100 * i for i in range(len(losses))], losses)
    plt.xlabel('n_epochs')
    plt.ylabel('mean_loss')
    plt.yscale('log')
    #plt.ylim(10 ** -6, 10 ** -2)
    plt.title(figure_name)
    try:
        os.makedirs("result_plots")
    except FileExistsError:
        pass
    print(figure_name)
    plt.savefig(f"result_plots/{figure_name}.png")


def one_experiment(cfg):
    torch.manual_seed(10)
    random.seed(10)
    np.random.seed(10)
    N = cfg['N'][0]
    d = cfg['d'][0]
    n_W = cfg['n_W'][0]
    masking = cfg['masking'][0]
    with_g = cfg['with_g'][0]

    data, labels = generate_3class_data(n_data=cfg['n_data'][0], n_vec=N, n_dim=d)
    trainset = DatasetCreator(
        mode='train',
        data=data,
        labels=labels
    )
    data_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg['batch_size'][0],
        shuffle=True
    )
    net = InteractionModule(cfg['n_class'][0], n_W, N, d, cfg['n_hidden_f'][0],
                            cfg['n_hidden_g'][0], masking, with_g)
    net.apply(weights_init)
    print(net)
    loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), cfg['LR'][0])

    # Training
    test_freq = 100
    losses = []
    for epoch in range(cfg['num_epoch'][0]):
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
            mean_loss = total_loss / cfg['n_data'][0]
            print("epoch=%d/%d, mean_loss=%.10f" % (epoch, cfg['num_epoch'][0], mean_loss))
            losses.append(float(mean_loss))
            net.train()
            if mean_loss < 0.0002:
                optimizer = optim.Adam(net.parameters(), lr=1e-5)

    plot_metrics(losses, N, d, n_W, masking, with_g)

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


if __name__ == '__main__':
    cfg = {
        'n_data': [12000],
        'num_epoch': [300],
        'LR': [1e-3],
        'batch_size': [120],
        'n_class': [3],
        'n_hidden_f': [20],
        'n_hidden_g': [3]
    }
    cfg_params = {
        'N': [16, 64, 256],
        'd': [6, 12, 128],
        'with_g': [False, True],
        'masking': [False, True]
    }
    for N in cfg_params['N']:
        for d in cfg_params['d']:
            for with_g in cfg_params['with_g']:
                for masking in cfg_params['masking']:
                    n_W = int(math.log(N, 2))
                    for n in [n_W, n_W+1]:
                        cfg['N'] = [N]
                        cfg['d'] = [d]
                        cfg['n_W'] = [n]
                        cfg['with_g'] = [with_g]
                        cfg['masking'] = [masking]
                        one_experiment(cfg)
