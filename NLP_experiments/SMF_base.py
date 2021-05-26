import torch
import torch.nn as nn
from ChordMatrix import chord_mask
device = torch.device('cuda')


def weights_init(module):
    if type(module) == torch.nn.Linear:
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


class VIdenticalModule(nn.Module):
    def __init__(self):
        super(VIdenticalModule, self).__init__()

    def forward(self, data):
        return data


class InteractionModule(nn.Module):
    def __init__(self, n_class, n_W, n_vec, n_dim, n_hidden_f=10, n_hidden_g=10, masking=False, with_g = False):
        super(InteractionModule, self).__init__()
        self.n_vec = n_vec
        self.n_dim = n_dim
        self.chord_mask = chord_mask(self.n_vec)
        self.masking = masking
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        if not with_g:
            self.g = VIdenticalModule()
        else:
            self.g = VModule(n_dim, n_hidden_g)
            #print(self.g)
        self.final = nn.Linear(self.n_vec*self.n_dim, n_class, bias=True)
#         self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        V = self.g(data)
        for f in self.fs[::-1]:
            if self.masking:
                W = f(data) * self.chord_mask
            else:
                W = f(data)
            V = W @ V
        V = self.final(V.view(data.size(0), -1))
        return V


class InteractionModuleEmbed(nn.Module):
    def __init__(self, embedding_matrix, n_class, n_W, n_vec, n_dim, n_hidden_f=10, n_hidden_g=10, masking=False, with_g = False):
        super(InteractionModuleEmbed, self).__init__()
        self.n_vec = n_vec
        num_words = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim = embedding_dim
        )
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        self.embedding.weight.requires_grad = False
        self.n_dim = n_dim
        self.chord_mask = chord_mask(self.n_vec)
        self.masking = masking
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        if not with_g:
            self.g = VIdenticalModule()
        else:
            self.g = VModule(n_dim, n_hidden_g)
            #print(self.g)
        self.final = nn.Linear(self.n_vec*self.n_dim, n_class, bias=True)
#         self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        X = self.embedding(data).to(device)
        V = self.g(X)
        for f in self.fs[::-1]:
            if self.masking:
                W = f(X).to(device) * self.chord_mask.to(device)
            else:
                W = f(X).to(device)
            V = (W @ V).to(device)
        V = self.final(V.view(X.size(0), -1)).to(device)
        return V


class InteractionModuleEmbedSkip(nn.Module):
    def __init__(self, embedding_matrix, n_class, n_W, n_vec, n_dim, n_hidden_f=10, n_hidden_g=10, masking=True, with_g=True, residual_every=True):
        super(InteractionModuleEmbedSkip, self).__init__()
        self.n_vec = n_vec
        num_words = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim = embedding_dim
        )
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        self.embedding.weight.requires_grad = False
        self.n_dim = n_dim
        self.chord_mask = chord_mask(self.n_vec)
        self.masking = masking
        self.fs = nn.ModuleList(
            [WModule(n_vec, n_dim, n_hidden_f) for i in range(n_W)]
        )
        if not with_g:
            self.g = VIdenticalModule()
        else:
            self.g = VModule(n_dim, n_hidden_g)
            #print(self.g)
        self.final = nn.Linear(self.n_vec*self.n_dim, n_class, bias=True)
#         self.softmax = nn.Softmax(dim=1)
        self.residual_every = residual_every

    def forward(self, data):
        X = self.embedding(data).to(device)
        V = self.g(X)
        residual = V
        for f in self.fs[::-1]:
            if self.masking:
                W = f(X).to(device) * self.chord_mask.to(device)
            else:
                W = f(X).to(device)
            V = (W @ V).to(device)
            V += residual
            if self.residual_every:
                residual = V
        V = self.final(V.view(X.size(0), -1)).to(device)
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
        self.chord_mask = chord_mask(self.n_vec)

    def forward(self, data):
        V = self.g(data)
        residual = V
        for f in self.fs[::-1]:
            W = f(data)
            if self.mask_:
                W = W * self.chord_mask
            V = W @ V
            V += residual
            if self.residual_every:
                residual = V
        V = self.final(V.view(data.size(0), -1))
        return V


if __name__ == '__main__':
    net = InteractionModule(3, 4, 16, 6, 10, 10, True, False)
    print(net)
