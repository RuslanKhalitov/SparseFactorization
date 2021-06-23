import torch
import torch.nn as nn
from ChordMatrix import chord_mask
from transformers import BertModel
device = torch.device('cuda')
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


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


class InteractionModuleBert(nn.Module):
    def __init__(self, n_class, n_W, n_vec, n_dim, n_hidden_f=10, n_hidden_g=10, masking=True, with_g = True, residual_every=False):
        super(InteractionModuleBert, self).__init__()
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
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.residual_every = residual_every

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        data = outputs["last_hidden_state"]
        #print(data.shape)
        V = self.g(data)
        residual = V
        for f in self.fs[::-1]:
            if self.masking:
                W = f(data) * self.chord_mask
            else:
                W = f(data)
            V = W @ V
            V += residual
            if self.residual_every:
                residual = V
        V = self.drop(self.final(V.view(data.size(0), -1)))
        return V


if __name__ == '__main__':
    net = InteractionModuleBert(2, 8, 256, 768, 128, 128)
    print(net)
