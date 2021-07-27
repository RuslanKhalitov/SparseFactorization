import os
import random
import numpy as np
import torch
import itertools
from torch import nn, optim
from torch_sparse import spmm
import math
torch.manual_seed(10)
random.seed(10)
np.random.seed(10)

device = torch.device('cuda')


def get_chord_indices_assym(n_vec, n_link):
    """
    Generates the position indicies, based on the asymmetric Chord protocol (incl. itself).

    :param n_vec: number of vectors (i.e. length of a sequence)
    :param n_link: number of links in the Chord protocol
    :return: target indices in two lists, each is of size n_vec * (n_link + 1)
    """

    rows = list(
        itertools.chain(
            *[
                [i for j in range(n_link + 1)] for i in range(n_vec)
            ]
        )
    )

    cols = list(
        itertools.chain(
            *[
                [i] + [(i + 2 ** k) % n_vec for k in range(n_link)] for i in range(n_vec)
            ]
        )
    )
    return rows, cols


# def weights_init(module):
#     classname = module.__class__.__name__
#     if classname.find('Linear') != -1:
#         torch.nn.init.normal_(module.weight, 0.0, 1e-2)
#         if hasattr(module, 'bias') and module.bias is not None:
#             torch.nn.init.normal_(module.bias, 0.0, 1e-2)


# SMF with sparse multiplication
class VIdenticalModule(nn.Module):
    def __init__(self):
        super(VIdenticalModule, self).__init__()

    def forward(self, data):
        return data


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


class WModuleSparse(nn.Module):
    def __init__(self, n_link, n_dim, n_hidden):
        super(WModuleSparse, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_link + 1)
        )

    def forward(self, data):
        return self.network(data)

        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class InteractionModuleSparseChar(nn.Module):
    def __init__(self, char_vocab, embedding_dim, n_class, n_W, n_vec, n_dim, n_link,
                 n_hidden_f, n_hidden_v, batch_size, masking=True, with_g=True, residual_every=True):
        super(InteractionModuleSparseChar, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=char_vocab,
            embedding_dim=embedding_dim
        )
        self.dropout1 = 0.5
        self.dropout2 = nn.Dropout(0.8)
        self.pos_embedding = PositionalEncoding(embedding_dim, self.dropout1, n_vec)
        self.n_vec = n_vec
        self.n_dim = n_dim
        self.n_link = n_link
        self.batch_size = batch_size
        self.residual_every = residual_every
        self.fs = nn.ModuleList(
            [WModuleSparse(n_link, n_dim, n_hidden_f) for i in range(n_W)]
        )
        if not with_g:
            self.g = VIdenticalModule()
        else:
            self.g = VModule(n_dim, n_hidden_v)
        self.final = nn.Linear(n_vec*n_dim, n_class, bias=True)
        self.chord_indicies = torch.tensor(get_chord_indices_assym(n_vec, n_link))
        self.chord_indicies = self.chord_indicies.to(device)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.requires_grad = True

    def forward(self, X):
        X = self.embedding(X).to(device)
        data = self.pos_embedding(X)
        #print(X.shape)
        V = self.g(data)
        residual = V
        for f in self.fs[::-1]:
            W = f(data).to(device)

            V = spmm(
                self.chord_indicies,
                W.reshape(W.size(0), W.size(1) * W.size(2)),
                self.n_vec,
                self.n_vec,
                V
            )

            V += residual
            if self.residual_every:
                residual = V
        V = self.dropout2(V)
        V = self.final(V.view(X.size(0), -1))
        return V


if __name__ == "__main__":
    cfg = {
        'n_class': [4],
        'n_hidden_f': [20],
        'n_hidden_g': [20],
        'N': [128],
        'd': [100],
        'n_link': [8],
        'n_W': [8],
        'batch_size': [16],
        'with_g': [True],
        'masking': [True],
        'residual': [False]
    }
