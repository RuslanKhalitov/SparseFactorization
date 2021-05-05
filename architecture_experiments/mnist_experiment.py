import os
import random
import numpy as np
import pandas as pd
import torch
import time
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

download = True  # CHANGE IF YOU'VE ALREADY DOWNLOADED IT

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST(
    './train',
    download=download,
    train=True,
    transform=transform
)
valset = datasets.MNIST(
    './val',
    download=download,
    train=False,
    transform=transform
)


def seed_everything(seed=1234):
    """
    Fixes random seeds, to get reproducible results.
    :param seed: a random seed across all the used packages
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Simple MLP
def simpleMLP():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    net = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.GELU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.GELU(),
                        nn.Linear(hidden_sizes[1], output_size)
                        )
    return net


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
    def __init__(self, n_class, n_W, n_vec, n_dim, n_hidden_f=32, n_hidden_g=10):
        super(InteractionModule, self).__init__()
        self.n_vec = n_vec
        self.n_dim = n_dim
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        self.g = VIdenticalModule()
        self.final = nn.Linear(self.n_vec * self.n_dim, n_class, bias=True)

    def forward(self, data):
        V = self.g(data)
        for f in self.fs[::-1]:
            W = f(data)
            V = W @ V
        V = self.final(V.view(data.size(0), -1))
        return V


class InteractionModuleSkip(nn.Module):
    def __init__(self, n_class, n_W, n_vec, n_dim, n_hidden_f=32, n_hidden_g=10, residual_every=True):
        super(InteractionModuleSkip, self).__init__()
        self.n_vec = n_vec
        self.n_dim = n_dim
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        self.g = VIdenticalModule()
        self.final = nn.Linear(self.n_vec * self.n_dim, n_class, bias=True)
        self.residual_every = residual_every

    def forward(self, data):
        V = self.g(data)
        residual = V
        for f in self.fs[::-1]:
            W = f(data)
            V = W @ V
            V += residual
            if self.residual_every:
                residual = V
        V = self.final(V.view(data.size(0), -1))
        return V


def TrainSimpleMLP(
        net,
        trainloader,
        valloader,
        n_epochs,
        test_freq,
        optimizer,
        loss
):
    for epoch in range(n_epochs):
        # Training
        running_loss = 0
        for i, (X, Y) in enumerate(trainloader):
            optimizer.zero_grad()
            X = X.view(X.shape[0], -1)
            pred = net(X)
            output = loss(pred, Y)
            output.backward()
            optimizer.step()
            running_loss += output.item()

        print("Epoch {} - Training loss:   {}".format(epoch, running_loss / len(trainloader)))

        # Validation
        if epoch % test_freq == 0:
            net.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i, (X, Y) in enumerate(valloader):
                    X = X.view(X.shape[0], -1)
                    pred = net(X)
                    val_loss += loss(pred, Y)
            print("Epoch {} - Validation loss: {}".format(epoch, val_loss / len(valloader)))
            print('_' * 40)
            net.train()


def TrainSMF(
        net,
        trainloader,
        valloader,
        n_epochs,
        test_freq,
        optimizer,
        loss
):
    for epoch in range(n_epochs):
        # Training
        running_loss = 0
        for i, (X, Y) in enumerate(trainloader):
            optimizer.zero_grad()
            pred = net(X.squeeze())
            output = loss(pred, Y)
            output.backward()
            optimizer.step()
            running_loss += output.item()

        print("Epoch {} - Training loss:   {}".format(epoch, running_loss / len(trainloader)))

        # Validation
        if epoch % test_freq == 0:
            net.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i, (X, Y) in enumerate(valloader):
                    pred = net(X.squeeze())
                    val_loss += loss(pred, Y)
            print("Epoch {} - Validation loss: {}".format(epoch, val_loss / len(valloader)))
            print('_' * 40)
            net.train()


if __name__ == '__main__':
    seed_everything()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    loss = nn.CrossEntropyLoss()
    n_epochs = 30
    test_freq = 1

    # MLP training
    net = simpleMLP()

    print('Simple MLP')
    TrainSimpleMLP(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        n_epochs=n_epochs,
        test_freq=test_freq,
        optimizer=optim.Adam(net.parameters(), lr=1e-3),
        loss=loss
    )

    # SMF Training
    n_classes = 10
    n_W = 4
    n_vec = 28
    n_dim = 28
    n_hidden_f = 32
    n_hidden_g = 3

    # No Skip Connections
    print('SMF without Skip Connections')
    net = InteractionModule(
        n_classes,
        n_W,
        n_vec,
        n_dim,
        n_hidden_f,
        n_hidden_g
    )
    net.apply(weights_init)

    TrainSMF(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        n_epochs=n_epochs,
        test_freq=test_freq,
        optimizer=optim.Adam(net.parameters(), lr=1e-3),
        loss=loss
    )

    # Skip Connections
    print('SMF with Skip Connections')
    net = InteractionModuleSkip(
        n_classes,
        n_W,
        n_vec,
        n_dim,
        n_hidden_f,
        n_hidden_g,
        residual_every=False
    )
    net.apply(weights_init)

    TrainSMF(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        n_epochs=n_epochs,
        test_freq=test_freq,
        optimizer=optim.Adam(net.parameters(), lr=1e-3),
        loss=loss
    )
