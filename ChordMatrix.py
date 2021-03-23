"""
Author: ShuttleNet team (NTNU)
"""

import numpy as np


def chord_mask(N, base=2, self_loop=True):
    ts = int(round(np.log(N)/np.log(base))) + 1

    if self_loop:
        ch = np.eye(N)
    else:
        ch = np.zeros(N)

    for i in range(N):
        for t in range(ts):
            # ch[i, abs((i + base ** t)) % N] = 1
            ch[i, ((i - 1) + base ** t) % N] = 1
    return ch


# print(chord_mask(12, 2, True))
print(chord_mask(17, 2, True))



