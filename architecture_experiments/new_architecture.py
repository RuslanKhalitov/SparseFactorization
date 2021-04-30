import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
import time
import multiprocessing
from typing import List, Dict

# Importing local supplementary files
from SMF_torch_deep import *
from torch_test import *
from Analysis import Analysis, PlotGraphs

cfg: Dict[str, List[int]] = {
    # 'folder_name': ['generate_permute_data_gaussian'],
    'folder_name': ['generate_exp_data'],
    'f': [13, 10],
    'g': [13, 10],
    'n_layers': [6],
    'N': [64],
    'd': [12],
    'disable_masking': [True],
    'LR': [0.001],
    'optimizer': ['Adam'],
    'batch_size': [200],
    'n_epochs': 200
}


class TSMF_1(SMFNet):
    def forward(self, X):
        V0 = self.g(X.float())
        # print('V0', np.round(V0.detach().numpy(), 1))
        for m in range(len(self.fs)):
            if self.disable_masking:
                W = self.fs[m](X.float())
            else:
                W = self.fs[m](X.float()) * self.chord_mask
            # print('Chord', np.round(self.chord_mask.detach().numpy(), 1))
            # print(f'W{m}', np.round(W.detach().numpy(), 1))
            V0 = torch.matmul(W, V0)
        return V0


class TSMF_2(SMFNet):
    def forward(self, X):
        V0 = self.g(X.float())
        # print('V0', np.round(V0.detach().numpy(), 1))
        for m in range(len(self.fs)):
            if m % 2 != 0:
                if self.disable_masking:
                    W = self.fs[m](X.float())
                else:
                    W = (self.fs[m](X.float()) * self.chord_mask).T
            else:
                if self.disable_masking:
                    W = self.fs[m](X.float())
                else:
                    W = self.fs[m](X.float()) * self.chord_mask
            # print('Chord', np.round(self.chord_mask.detach().numpy(), 1))
            # print(f'W{m}', np.round(W.detach().numpy(), 1))
            V0 = torch.matmul(W, V0)
        return V0


class TSMF_3(SMFNet):
    def forward(self, X):
        V0 = self.g(X.float())
        # print('V0', np.round(V0.detach().numpy(), 1))
        for m in range(len(self.fs)):
            if m % 2 == 0:
                if self.disable_masking:
                    W = self.fs[m](X.float())
                else:
                    W = (self.fs[m](X.float()) * self.chord_mask).T
            else:
                if self.disable_masking:
                    W = self.fs[m](X.float())
                else:
                    W = self.fs[m](X.float()) * self.chord_mask
            # print('Chord', np.round(self.chord_mask.detach().numpy(), 1))
            # print(f'W{m}', np.round(W.detach().numpy(), 1))
            V0 = torch.matmul(W, V0)

        return V0


def Changed_SMF_full(cfg: Dict[str, List]) -> TSMF_1:
    model = TSMF_3(
        g=make_layers_g(cfg),
        fs=nn.ModuleList(
            [make_layers_f(cfg) for _ in range(cfg['n_layers'][0])]
        ),
        N=cfg['N'][0],
        disable_masking=cfg['disable_masking'][0]
    )
    return model

#Train
def one_experient(cfg):

    seed_everything(1234)
    assert str(os.getcwd()) == YOUR_DIRECTORY_NAME,\
        "Please specify parameter YOUR_DIRECTORY_NAME"

    model = Changed_SMF_full(cfg)
    print(model)

    criterion = torch.nn.MSELoss(reduction='mean')

    # Mapping optimizer
    optim_mapped = {
        'Adam': torch.optim.Adam,
        'RMSP': torch.optim.RMSprop,
        'SGD': torch.optim.SGD
    }[cfg['optimizer'][0]]

    optimizer = optim_mapped(model.parameters(), lr=cfg['LR'][0])
    train_loader, test_loader = load_data(cfg['batch_size'][0], cfg['folder_name'][0])

    final_dict = {
        'train_loss': [],
        'test_loss': [],

        'grad_g_value': [],
        'grad_g_bias': [],
        'grad_f_value': [],
        'grad_f_bias': [],

        'g_weight_std': [],
        'g_weight_mean': [],
        'g_weight_max': [],

        'g_bias_std': [],
        'g_bias_mean': [],
        'g_bias_max': [],

        'fs_weight_std': [],
        'fs_weight_mean': [],
        'fs_weight_max': [],

        'fs_bias_std': [],
        'fs_bias_mean': [],
        'fs_bias_max': [],
    }

    for epoch in range(1, cfg['n_epochs'] + 1):

        loss, loss_ev = train(model, epoch, train_loader, test_loader, criterion, optimizer)
        final_dict['train_loss'].append(loss)
        final_dict['test_loss'].append(loss_ev)

        final_dict = Analysis(model, cfg, final_dict).stats_on_params()

        if epoch == cfg['n_epochs'] - 1:
            try:
                os.makedirs("SparseFactorization/output/model")
            except FileExistsError:
                pass
            torch.save(model.state_dict(), "SparseFactorization/output/model/final_model.pth")
        # print(final_dict)
    PlotGraphs(final_dict, cfg).plot()

# one_experient(cfg)

model = Changed_SMF_full(cfg)
model.load_state_dict(torch.load('SparseFactorization/output/model/final_model.pth'))
model.eval()

X = np.genfromtxt('SparseFactorization/train/{}/X/X_0.csv'.format(cfg['folder_name'][0]), delimiter=',')
print('X', np.round(X, 1))
X_gt = model(torch.from_numpy(X)).detach().numpy()
print('X_gt', np.round(X_gt, 1))
Y = np.genfromtxt('SparseFactorization/train/{}/Y/Y_0.csv'.format(cfg['folder_name'][0]), delimiter=',')
print('Y', np.round(Y, 1))