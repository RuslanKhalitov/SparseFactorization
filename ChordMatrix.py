"""
Author: ShuttleNet team (NTNU)
"""

import numpy as np


def chord_mask(N, base=2):
    ts = int(round(np.log(N)/np.log(base)))
    ch = np.eye(N)
    for i in range(N):
        for t in range(ts):
            ch[i, abs((i + 2 ** t)) - N] = 1
    return ch


