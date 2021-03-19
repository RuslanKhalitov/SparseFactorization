"""
Authors: ShuttleNet team (NTNU)

Parametric SF.
This module is going to be used for the concept proof on small matrices.
"""

import numpy as np
import torch
from ChordMatrix import chord_mask
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SparseFactorization(nn.Module):
    def __init__(self,
                 chord_base,
                 embedding_size,
                 src_vocab_size,
                 max_len,
                 n_layers,
                 bias,
                 device
                 ):
        super(SparseFactorization, self).__init__()
        self.chord_base = chord_base
        self.n_layers = n_layers
        self.max_len = max_len
        self.device = device
        self.bias = bias
        self.obj_embedding = nn.Embedding(src_vocab_size, embedding_size)
        # self.pos_embedding = nn.Embedding(max_len, embedding_size)
        self.g = nn.Linear(embedding_size, max_len, bias=self.bias)

        # ModuleList can be indexed like a regular Python list, but modules it contains
        # are properly registered, and will be visible by all Module methods.
        self.fm = nn.ModuleList(
            [
                nn.Linear(N, max_len * max_len, bias=self.bias)
                for _ in range(n_layers)
            ]
        )

    def make_W_mask(self, N):
        trg_mask = chord_mask(N, self.chord_base)
        trg_mask = torch.flatten(trg_mask)
        return trg_mask.to(self.device)

    def forward(self, x, mask=True):
        """
        Performs the forward path according to the initial design
        :param x: The input tensor — E(l)
        :param mask: Whether masking is required or not
        :return: E(l+1)
        """
        N, seq_length = x.shape
        assert seq_length == self.max_len, "Please pad input sequence before executing the modules"

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        if mask:
            mask = self.make_W_mask(seq_length)  # generate chord mask
        # full_embedding = self.obj_embedding(x) + self.pos_embedding(positions)
        full_embedding = self.obj_embedding(x)
        final = self.g(full_embedding)  # calculate V

        for F in self.fm:
            # linear module to calculate a chord matrix
            out = F(full_embedding)
            if mask:
                # masking according to the chord mask
                out = torch.masked_select(out, mask)
                out = out.view(-1, seq_length)  # make it squared
            final = torch.mm(final, out)  # multiplication

        assert final.shape == x.shape, "Output tensor should be of the same size as input tensor"
        return final


SF = SparseFactorization(2, 25, 100, 100, 3, False, device)
print(SF)
