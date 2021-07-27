import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.weight.requires_grad = True

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.ninp)
        #print(src.shape, src[0])
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.8, max_len=5000):
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


class Classifier(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, last_hidden, nclass, max_len):
        super(Classifier, self).__init__()
        self.transformer = TransformerModel(ntoken, ninp, nhead, nhid, nlayers)
        #self.pooling = nn.AvgPool1d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.8)
        #self.dense = nn.Linear(max_len*(int((ninp-3)/2)+1), last_hidden)# int((self.n_dim-3)/2)+1
        self.final = nn.Linear(max_len*ninp, nclass)

    def forward(self, x):

        x = self.transformer(x)
        x = self.dropout(x)
        x = self.final(x.view(x.size(0), -1))
        return x
