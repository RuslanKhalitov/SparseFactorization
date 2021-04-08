"""
The design was taken from
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Dict, Any, cast

# GLobals
N = 16
d = 5
n_layers = 4


def make_chord(N):
    """
    :param N: Length of the original data
    :return: torch tensor
    """
    chord_mask = torch.eye(N)
    for i in range(N):
        for k in range(2):
            chord_mask[i][(i + np.power(2, k) - 1) % N] = 1

    return chord_mask


class SMFNet(nn.Module):
    def __init__(
            self,
            g: nn.Module,
            fs: nn.Module
    ) -> None:
        """
        Main architecture
        :param g: NN for F
        :param fs: NNs for Ws
        """
        super(SMFNet, self).__init__()

        self.g = g
        self.fs = fs

    def forward(self, X):
        """
        :param X: Nxd data
        :return: V(M-1) M is the layer number.
        """
        N = X.shape[1]
        chord_mask = make_chord(N)
        V0 = self.g(X.float())
        for m in range(len(self.fs)):
            W = self.fs[m](X.float())
            # W = self.fs[m](X.float()) * chord_mask
            V0 = torch.matmul(W, V0)

        return V0


def make_layers_f(cfg: List[Union[str, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_size = d
    for i in range(len(cfg) - 1):
        v = cast(int, cfg[i])
        linear = nn.Linear(in_size, v)
        layers += [linear, nn.ReLU(inplace=True)]
        in_size = v
    layers += [nn.Linear(in_size, cast(int, cfg[-1]))]
    return nn.Sequential(*layers)


def make_layers_g(cfg: List[Union[str, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_size = d
    for i in range(len(cfg) - 1):
        v = cast(int, cfg[i])
        linear = nn.Linear(in_size, v)
        layers += [linear, nn.ReLU(inplace=True)]
        in_size = v
    layers += [nn.Linear(in_size, cast(int, cfg[-1]))]
    return nn.Sequential(*layers)


def SMF_full(cfg: str) -> SMFNet:
    model = SMFNet(
        g=make_layers_g(cfg['g']),
        fs=nn.ModuleList(
            [make_layers_f(cfg['f']) for _ in range(n_layers)]
        )
    )
    return model


# if __name__ == '__main__':
#     model = SMF_full(cfg)
#     print(model)
