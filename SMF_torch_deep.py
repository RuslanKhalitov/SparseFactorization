"""
The design was taken from
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Dict, Any, cast


def make_chord(N):
    """
    :param N: Length of the original data
    :return: torch tensor
    """
    chord_mask = torch.eye(N, requires_grad=False)
    for i in range(N):
        for k in range(2):
            chord_mask[i][(i + np.power(2, k) - 1) % N] = 1

    return chord_mask


class SMFNet(nn.Module):
    def __init__(
            self,
            g: nn.Module,
            fs: nn.Module,
            N: int
    ) -> None:
        """
        Main architecture
        :param g: NN for F
        :param fs: NNs for Ws
        :param N: sequence length
        """
        super(SMFNet, self).__init__()
        self.N = N
        self.g = g
        self.fs = fs
        self.chord_mask = make_chord(self.N)

    def forward(self, X):
        """
        :param X: Nxd data
        :return: V(M-1) M is the layer number.
        """
        N = X.shape[1]
        V0 = self.g(X.float())
        for m in range(len(self.fs)):
            # W = self.fs[m](X.float())
            W = self.fs[m](X.float()) * self.chord_mask
            V0 = torch.matmul(W, V0)

        return V0


def make_layers_f(cfg: Dict[str, List]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_size = cast(int, cfg['d'][0])
    cfg_g = cfg['g']
    for i in range(len(cfg_g)):
        v = cast(int, cfg_g[i])
        linear = nn.Linear(in_size, v)
        layers += [linear, nn.ReLU(inplace=True)]
        in_size = v
    layers += [nn.Linear(in_size, cast(int, cfg['N'][0]))]
    return nn.Sequential(*layers)


def make_layers_g(cfg: Dict[str, List]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_size = cast(int, cfg['d'][0])
    cfg_f = cfg['f']
    for i in range(len(cfg_f)):
        v = cast(int, cfg_f[i])
        linear = nn.Linear(in_size, v)
        layers += [linear, nn.ReLU(inplace=True)]
        in_size = v
    layers += [nn.Linear(in_size, cast(int, cfg['d'][0]))]
    return nn.Sequential(*layers)


def SMF_full(cfg: Dict[str, List]) -> SMFNet:
    model = SMFNet(
        g=make_layers_g(cfg),
        fs=nn.ModuleList(
            [make_layers_f(cfg) for _ in range(cfg['n_layers'][0])]
        ),
        N=cfg['N'][0])
    return model


# if __name__ == '__main__':
#     model = SMF_full(cfg)
#     print(model)
