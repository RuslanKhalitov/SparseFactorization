import math
import sys
import numpy as np
import torch
import itertools
from torch import nn
from typing import Union, List

import torch_geometric
from torch_sparse import spmm
from torch.utils.data import Dataset


def get_chord_indices_assym(n_vec, n_link):
    """
    Generates position indicies, based on the Chord protocol (incl. itself).

    :param n_vec: sequence length
    :param n_link: number of links in the Chord protocol
    :return: target indices in two lists, each is of size n_vec * n_link
    """
    
    rows = list(
        itertools.chain(
            *[
                [i for j in range(n_link)] for i in range(n_vec)
            ]
        )
    )
    
    cols = list(
        itertools.chain(
            *[
                [i] + [(i + 2 ** k) % n_vec for k in range(n_link - 1)] for i in range(n_vec)
            ]
        )
    )
    
    return rows, cols
    

def MakeMLP(cfg: List[Union[str, int]], in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Constructs an MLP based on a given structural config. 
    """
    layers: List[nn.Module] = []
    for i in cfg:
        if isinstance(i, int):
            layers += [nn.Linear(in_channels, i)]
            in_channels = i
        else:
            layers += [nn.GELU()]
    layers += [nn.Linear(in_channels, out_channels)]
    return nn.Sequential(*layers)
    

class MLPBlock(nn.Module):
    """
    Constructs a MLP with the specified structure.
    
    """
    def __init__(self, cfg, in_dim, out_dim):
        super(MLPBlock, self).__init__()
        self.network = MakeMLP(cfg, in_dim, out_dim)

    def forward(self, data):
        return self.network(data)


class PSFNet(nn.Module):
    def __init__(self,
    vocab_size,
    embedding_size,
    max_seq_len,
    use_cuda,
    use_residuals,
    dropout1_p,
    dropout2_p,
    dropout3_p
    ):
        super(PSFNet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.n_W = math.ceil(np.log2(self.max_seq_len))
        self.n_links = self.n_W + 1
        self.Ws = [embedding_size, 'GELU']
        self.V = [embedding_size, 'GELU']
        self.use_cuda = use_cuda
        self.use_residuals = use_residuals
        self.dropout1_p = dropout1_p
        self.dropout2_p = dropout2_p
        self.dropout3_p = dropout3_p

        # Init Ws
        self.fs = nn.ModuleList(
            [
                MLPBlock(
                    self.Ws,
                    self.embedding_size,
                    self.n_links
                )
                for _ in range(self.n_W)
            ]
        )

        # Init V
        self.g = MLPBlock(
            self.V,
            self.embedding_size,
            self.embedding_size
        )
        
        self.dropout1 = nn.Dropout(self.dropout1_p)
        self.dropout2 = nn.Dropout(self.dropout2_p)
        self.dropout3 = nn.Dropout(self.dropout3_p)
        
        self.chord_indicies = torch.tensor(get_chord_indices_assym(self.max_seq_len, self.n_links))

        # Init embedding layer
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_size
        )

        # Init APC
        self.apc_embedding = nn.Embedding(
            self.max_seq_len,
            self.embedding_size
        )

    def forward(self, data):
        
        # Get embedding
        data = self.embedding(data)

        # Add APC
        positions = torch.arange(0, self.max_seq_len).expand(data.size(0), self.max_seq_len)
        if self.use_cuda:
            positions = positions.cuda()
        pos_embed = self.apc_embedding(positions)
        data = data + pos_embed
        
        # Apply the first dropout
        data = self.dropout1(data)

        # Get V 
        V = self.g(data)

        # Apply the second dropout
        V = self.dropout2(V)

        # Init residual connection if needed
        if self.use_residuals:
            res_conn = V

        # Iterate over all W
        for m in range(self.n_W):

            # Get W_m  
            W = self.fs[m](data)

            # Multiply W_m and V, get new V
            V = spmm(
                self.chord_indicies,
                W.reshape(W.size(0), W.size(1) * W.size(2)), 
                self.max_seq_len,
                self.max_seq_len,
                V
            )

            # Apply residual connection
            if self.use_residuals:
                V = V + res_conn

        # Apply the third dropout
        V = self.dropout3(V)
        return V 


net = PSFNet(
    vocab_size=256,
    embedding_size=128,
    max_seq_len=1024,
    use_cuda=True,
    use_residuals=False,
    dropout1_p=0,
    dropout2_p=0,
    dropout3_p=0
)

print(net)

# Example of DataLoader
class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.data[index]
        Y = self.labels[index].to(dtype=torch.long)
        return (X, Y)

    def __len__(self):
        return len(self.labels)


# trainset = DatasetCreator(
#     data = data,
#     labels = labels
# )

# trainloader = torch_geometric.data.DataLoader(
#     trainset,
#     batch_size=32,
#     shuffle=True,
#     drop_last=True
# )
