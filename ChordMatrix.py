"""
Author: ShuttleNet team (NTNU)
"""

import numpy as np
import torch


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


# print(chord_mask(17, 2, True))



