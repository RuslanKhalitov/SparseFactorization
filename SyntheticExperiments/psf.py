import torch
import itertools
from torch import nn
from typing import Union, List
from torch_sparse import spmm

def get_chord_indices_assym(n_vec, n_link):
    """
    Generates the position indicies, based on the Chord protocol (incl. itself).

    :param n_vec: number of vectors (i.e. length of a sequence)
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
    n_vec,
    n_W,
    Ws,
    V,
    n_channels_V,
    n_class,
    pooling_type,
    head,
    use_cuda,
    use_residuals,
    dropout1_p,
    dropout2_p,
    dropout3_p,
    init_embedding_weights,
    use_pos_embedding,
    problem):
        super(PSFNet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_vec = n_vec
        self.n_W = n_W
        self.n_links = n_W + 1
        self.Ws = Ws
        self.V = V
        self.n_channels_V = n_channels_V
        self.n_class = n_class
        self.pooling_type = pooling_type
        self.head = head
        self.use_cuda = use_cuda
        self.use_residuals = use_residuals
        self.dropout1_p = dropout1_p
        self.dropout2_p = dropout2_p
        self.dropout3_p = dropout3_p
        self.init_embedding_weights = init_embedding_weights
        self.use_pos_embedding = use_pos_embedding
        self.problem = problem

        # Init embedding layers
        # print(f'Making embedding for {self.problem}')
        if (self.problem == 'imdb') or (self.problem == 'listops'):
            self.embedding = nn.Embedding(
                self.vocab_size,
                self.embedding_size,
                padding_idx=self.vocab_size-2
            )
        elif (self.problem == 'cifar10') or (self.problem == 'pathfinder'):
            self.embedding = nn.Embedding(
                self.vocab_size,
                self.embedding_size
            )

        # Init positional embeding layer
        self.pos_embedding = nn.Embedding(
            self.n_vec,
            self.embedding_size
        )

        if self.init_embedding_weights:
            self.init_embed_weights()

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
            self.n_channels_V
        )
        
        # Init final layer
        if self.head[0] == 'linear':
            if self.pooling_type == 'FLATTEN':
                self.final =  nn.Linear(
                    self.n_vec * self.n_channels_V,
                    self.n_class
                )
            elif self.pooling_type == 'CLS':
                self.final =  nn.Linear(
                    self.n_channels_V,
                    self.n_class
                )
        elif self.head[0] == 'non-linear':
            if self.pooling_type == 'FLATTEN':
                self.final = nn.Sequential(
                    nn.Linear(
                        self.n_vec * self.n_channels_V,
                        self.head[1]
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.head[1],
                        self.n_class
                    )
                )
            elif self.pooling_type == 'CLS':
                self.final = nn.Sequential(
                    nn.Linear(
                        self.n_channels_V,
                        self.head[1]
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.head[1],
                        self.n_class
                    )
                )

        self.dropout1 = nn.Dropout(self.dropout1_p)
        self.dropout2 = nn.Dropout(self.dropout2_p)
        self.dropout3 = nn.Dropout(self.dropout3_p)
        
        self.chord_indicies = torch.tensor(get_chord_indices_assym(self.n_vec, self.n_links))
        if self.use_cuda:
            self.chord_indicies = self.chord_indicies.cuda()
    
    def init_embed_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.requires_grad = True

    def forward(self, data):
        
        # Get embedding
        data = self.embedding(data)

        # Get positional embedding if needed
        if self.use_pos_embedding:
            positions = torch.arange(0, self.n_vec).expand(data.size(0), self.n_vec)
            if self.use_cuda:
                positions = positions.cuda()
            pos_embed = self.pos_embedding(positions)
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
                self.n_vec,
                self.n_vec,
                V
            )
            
            # Apply residual connection
            if self.use_residuals:
                V = V + res_conn

        # Apply the third dropout
        V = self.dropout3(V)
            
        if self.pooling_type == 'CLS':
            V = V[:, 0, :]

        V = self.final(V.view(V.size(0), -1))
        return V 





