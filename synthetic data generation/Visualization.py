from two_class_data_generation import *

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
import seaborn as sns
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

download = False

def chord_mask(N, base=2, self_loop=True):
    ts = int(round(np.log(N)/np.log(base))) + 1

    if self_loop:
        ch = torch.eye(N, requires_grad=False)
    else:
        ch = torch.zeros(N, requires_grad=False)

    for i in range(N):
        for t in range(ts):
            ch[i, ((i - 1) + base ** t) % N] = 1
    return ch


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
    def __init__(self, n_class, n_W, n_vec, n_dim, n_hidden_f=32, n_hidden_g=10, mask_=True):
        super(InteractionModule, self).__init__()
        self.n_vec = n_vec
        self.n_dim = n_dim
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        self.g = VIdenticalModule()
        self.final = nn.Linear(self.n_vec * self.n_dim, n_class, bias=True)
        self.mask_ = mask_

    #     def forward(self, data):
    #         V = self.g(data)
    #         W_final = torch.eye(self.n_vec, self.n_vec)
    #         for f in self.fs[::-1]:
    #             W = f(data)
    #             if self.mask_:
    #                 W = W * masking
    #             W_final = W_final @ W
    #         V = W_final @ V
    #         V = self.final(V.view(data.size(0), -1))
    #         return V, W_final

    def forward(self, data):
        V = self.g(data)
        for f in self.fs[::-1]:
            W = f(data)
            if self.mask_:
                W = W * masking
            V = W @ V
        V = self.final(V.view(data.size(0), -1))
        return V


class InteractionModuleSkip(nn.Module):
    def __init__(self, n_class, n_W, n_vec, n_dim, n_hidden_f=32, n_hidden_g=10, residual_every=True, mask_=True):
        super(InteractionModuleSkip, self).__init__()
        self.n_vec = n_vec
        self.n_dim = n_dim
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        self.g = VIdenticalModule()
        self.final = nn.Linear(self.n_vec * self.n_dim, n_class, bias=True)
        self.residual_every = residual_every
        self.mask_ = mask_

    def forward(self, data):
        V = self.g(data)
        residual = V
        W_final = torch.eye(self.n_vec, self.n_vec)
        for idx, f in enumerate(self.fs[::-1]):
            W = f(data)
            if self.mask_:
                W = W * masking
            W_final = W_final @ W
            V = W @ V
            V += residual
            if self.residual_every:
                residual = V
        V = self.final(V.view(data.size(0), -1))
        return V, W_final


class InteractionModuleSkipLN(nn.Module):
    def __init__(self, n_class, n_W, n_vec, n_dim, n_hidden_f=32, n_hidden_g=10, residual_every=True, mask_=True):
        super(InteractionModuleSkip, self).__init__()
        self.n_vec = n_vec
        self.n_dim = n_dim
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        self.g = VIdenticalModule()
        self.final = nn.Linear(self.n_vec * self.n_dim, n_class, bias=True)
        self.residual_every = residual_every
        self.mask_ = mask_
        self.LNinit = nn.LayerNorm([self.n_vec, self.n_dim])
        self.LNend = nn.ModuleList([nn.LayerNorm([self.n_vec, self.n_dim]) for i in range(n_W)])

    def forward(self, data):
        data = self.LNinit(data)
        V = self.g(data)
        residual = V
        W_final = torch.eye(self.n_vec, self.n_vec)
        for idx, f in enumerate(self.fs[::-1]):
            W = f(data)
            if self.mask_:
                W = W * masking
            W_final = W_final @ W
            V = W @ V
            V += residual
            V = self.LNend[idx](V)
            if self.residual_every:
                residual = V
        V = self.final(V.view(data.size(0), -1))
        return V, W_final


class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, mode, data, labels):
        print(f'Creating data loader - {mode}')
        assert mode in ['train', 'test']
        self.data = data
        self.labels = labels
        assert len(self.data) == len(self.labels), \
            "The number of samples doesn't match the number of labels"

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.data[index].unsqueeze(-1)
        Y = self.labels[index].type(torch.LongTensor)
        return (X, Y)

    def __len__(self):
        return len(self.labels)


def visualize(data, W, epoch):
    data = data.detach().numpy()
    fig, axes = plt.subplots(nrows=len(W), ncols=2, figsize=(15, 45), facecolor="w")
    for i in range(len(W)):
        axes[i][0].plot(data[i], '.')
        axes[i][0].set_title('Epoch: {}, sample: {}'.format(epoch, i))
        axes[i][0].set_ylim(data[i].min()-.3, data[i].max()+.3)
        axes[i][1].imshow(W[i].detach().numpy(), cmap="Blues", interpolation='nearest')
        tight_layout()
    savefig(f'./W_epoch_{epoch}.png')


def TrainSMF(
        net,
        trainloader,
        valloader,
        n_epochs,
        test_freq,
        optimizer,
        loss
):
    losses = []
    losses_eval = []
    accuracies = []
    for epoch in range(n_epochs):
        # Training
        running_loss = 0

        for i, (X, Y) in enumerate(trainloader):
            optimizer.zero_grad()
            pred, W_final = net(X)
            #             pred = net(X)
            output = loss(pred, Y)
            output.backward()
            optimizer.step()
            running_loss += output.item()

        if epoch % 3 == 0:
            vis_dict = {
                'X': X,
                'labels': Y,
                'W_final': W_final
            }
            visualize(vis_dict['X'], vis_dict['W_final'], epoch)

        print("Epoch {} - Training loss:   {}".format(epoch, running_loss / len(trainloader)))
        losses.append(float(running_loss / len(trainloader)))

        # Validation
        if epoch % test_freq == 0:
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0.0
                for i, (X, Y) in enumerate(valloader):
                    pred, Vs = net(X)
                    #                     pred = net(X)
                    val_loss += loss(pred, Y).item()

                    _, predicted = torch.max(pred.data, 1)
                    #                     print('prediction', pred.data)
                    #                     print('pred max', predicted)
                    total += Y.size(0)
                    correct += (predicted == Y).sum().item()

            print("Epoch {} - Validation loss: {}".format(epoch, val_loss / len(valloader)))
            print(f'Correct: {correct}, all: {total}')
            print('Accuracy of the network: %d %%' % (100 * correct / total))
            print('_' * 40)
            losses_eval.append(float(val_loss / len(valloader)))
            accuracies.append(100 * correct / total)
            net.train()

    fig, axs = plt.subplots(1, 2)
    axs[0].plot([i for i in range(len(losses))], losses, label="Train")
    axs[0].plot([i for i in range(len(losses_eval))], losses_eval, label="Validation")
    axs[0].set_xlabel('n_epochs')
    axs[0].set_ylabel('mean_loss')
    #     axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot([i for i in range(len(accuracies))], accuracies)
    axs[1].set_xlabel('n_epochs')
    axs[1].set_ylabel('test accuracy')
    #     axs[1].set_yscale('log')
    savefig('./SMF_accuracy.png')


torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

n_data = 800
n_test = 400
n_class = 2
n_W = 10
n_vec = 1024
n_dim = 1
n_hidden_f = 64
n_hidden_g = 3
batch_size = 8

masking = chord_mask(n_vec)

# data, labels, data_orig, interaction, mixing = generate_two_class_mixed_data(
#     n_data,
#     n_vec,
#     binary=False,
#     same_sigma=False,
#     xor=True
# )

data, labels = generate_two_class_data(n_data, n_vec, binary=False, same_sigma=False, xor=True)
# Train / Test split
dataset_size = len(data)
indices = list(range(dataset_size))
split = n_test
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

data, labels, data_val, labels_val = data[train_indices], labels[train_indices], data[val_indices], labels[val_indices]
print(len(data))
print(len(data_val))

trainset = DatasetCreator(
    mode='train',
    data = data,
    labels = labels
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True
)


valset = DatasetCreator(
    mode='test',
    data = data_val,
    labels = labels_val
)
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=batch_size,
    shuffle=True
)

# net = InteractionModule(n_class, n_W, n_vec, n_dim, n_hidden_f, n_hidden_g)
# net = InteractionModuleSkip(n_class, n_W, n_vec, n_dim, n_hidden_f, n_hidden_g)
net = InteractionModuleSkipLN(n_class, n_W, n_vec, n_dim, n_hidden_f, n_hidden_g)
net.apply(weights_init)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

TrainSMF(
    net=net,
    trainloader=trainloader,
    valloader=valloader,
    n_epochs=10,
    test_freq=3,
    optimizer=optimizer,
    loss=loss
)





