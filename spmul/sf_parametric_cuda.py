"""
An implementation of Interaction Network with Sparse Factorization (on CUDA)
This version uses symmetric Chord protocol, that is, the j-th node is linked to both j - 2 ^ k and j + 2 ^ k
"""
import torch
import torch.nn as nn
from models.spmul import *

device = torch.device('cuda')


class FNet(nn.Module):
    def __init__(self, n_dim, n_link_all, n_hidden_f):
        super(FNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_dim, n_hidden_f),
            nn.GELU(),
            nn.Linear(n_hidden_f, n_link_all)
        )

    def forward(self, inputs):
        return self.net(inputs)


class VNet(nn.Module):
    def __init__(self, n_dim, n_hidden_v):
        super(VNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_dim, n_hidden_v),
            nn.GELU(),
            nn.Linear(n_hidden_v, n_dim)
        )

    def forward(self, inputs):
        return self.net(inputs)


class InteractionNet(nn.Module):
    def __init__(self, n_vec, n_link_all, n_W, n_dim, n_hidden_f, n_hidden_v):
        super(InteractionNet, self).__init__()
        self.n_W = n_W
        self.fnets = nn.ModuleList([FNet(n_dim, n_link_all, n_hidden_f) for _ in range(n_link_all)])
        self.spmul = SparseMultiply.apply
        self.vnet = VNet(n_dim, n_hidden_v)
        self.offsets = torch.tensor([0] + [2 ** k for k in range(n_link_all - 1)], dtype=int, requires_grad=False).cuda()
        self.n_block = 16
        self.n_thread_vec = 64
        self.n_thread_dim = 16
        self.n_thread_link = 16

    def forward(self, inputs):
        Fs = [self.fnets[i](inputs) for i in range(self.n_W)]
        V = self.vnet(inputs)
        Z = V
        for k in range(self.n_W):
            Z = self.spmul(Fs[k], Z, self.offsets, self.n_block, self.n_thread_vec, self.n_thread_dim,
                           self.n_thread_link)
        return Z


class InteractionNetEmbed(nn.Module):
    def __init__(self, embedding_matrix, n_class, n_vec, n_link_all, n_W, n_dim, n_hidden_f, n_hidden_v):
        super(InteractionNetEmbed, self).__init__()
        num_words = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embedding_dim
        )
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.n_W = n_W
        self.fnets = nn.ModuleList([FNet(n_dim, n_link_all, n_hidden_f) for _ in range(n_link_all)])
        self.spmul = SparseMultiply.apply
        self.vnet = VNet(n_dim, n_hidden_v)
        self.offsets = torch.tensor([0] + [2 ** k for k in range(n_link_all - 1)], dtype=int, requires_grad=False).cuda()
        self.n_block = 16
        self.n_thread_vec = 64
        self.n_thread_dim = 16
        self.n_thread_link = 16
        self.final = nn.Linear(n_vec * n_dim, n_class, bias=True)

    def forward(self, inputs):
        X = self.embedding(inputs).to(device)
        Fs = [self.fnets[i](X) for i in range(self.n_W)]
        V = self.vnet(X)
        Z = V
        residual = V
        for k in range(self.n_W):
            Z = self.spmul(Fs[k], Z, self.offsets, self.n_block, self.n_thread_vec, self.n_thread_dim,
                           self.n_thread_link)
            Z += residual
        Z = self.final(V.view(inputs.size(0), -1).to(device))
        return Z
