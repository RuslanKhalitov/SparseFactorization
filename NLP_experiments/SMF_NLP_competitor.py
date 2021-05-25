import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        """
        Given embedding_matrix: numpy array with vector for all words
        return prediction ( in torch tensor format)
        """
        super(LSTM, self).__init__()
        # Number of words = number of rows in embedding matrix
        num_words = embedding_matrix.shape[0]
        # Dimension of embedding is num of columns in the matrix
        embedding_dim = embedding_matrix.shape[1]
        # Define an input embedding layer
        self.embedding = nn.Embedding(
                                      num_embeddings=num_words,
                                      embedding_dim=embedding_dim)
        # Embedding matrix actually is collection of parameter
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float32))
        # Because we use pretrained embedding (GLove, Fastext,etc) so we turn off requires_grad-meaning we do not train gradient on embedding weight
        self.embedding.weight.requires_grad = False
        # LSTM with hidden_size = 128
        self.lstm = nn.LSTM(
                            embedding_dim,
                            128,
                            bidirectional=True,
                            batch_first=True,
                             )
        # Input(512) because we use bi-directional LSTM ==> hidden_size*2 + maxpooling **2  = 128*4 = 512, will be explained more on forward method
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        # pass input (tokens) through embedding layer
        x = self.embedding(x)
        # fit embedding to LSTM
        hidden, _ = self.lstm(x)
        # apply mean and max pooling on lstm output
        avg_pool= torch.mean(hidden, 1)
        max_pool, index_max_pool = torch.max(hidden, 1)
        # concat avg_pool and max_pool ( so we have 256 size, also because this is bidirectional ==> 256*2 = 512)
        out = torch.cat((avg_pool, max_pool), 1)
        # fit out to self.out to conduct dimensionality reduction from 512 to 1
        out = self.out(out)
        # return output
        return out


class Attention(nn.Module):
    def __init__(self, input_dim, step_dim, bias=True):
        """
        The input is the matrice input_dim*step_dim where the input_dim is
        the dimension of hidden states and step_dim is number of units LSTM
        :param input_dim: the dimension of hidden stat 256
        :param step_dim: step_dim is number of units LSTM maxlen
        :param bias:
        """
        super(Attention, self).__init__()
        self.mask = True
        self.input_dim = input_dim
        self.step_dim = step_dim
        weight = torch.zeros(input_dim, 1)
        nn.init.kaiming_uniform(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(step_dim))

    def forward(self, h, mask=None):
        #eij = a(si-1 , hj) weight: 256*1
        # dim: (maxlen*256*256*1)-->maxlen*1-->1*maxlen
        e = torch.mm(h.view(-1, self.input_dim), self.weight).view(-1, self.step_dim)
        e = e + self.bias
        e = torch.tanh(e)
        # a = exp(eij)/sum(exp(eij))
        # dim: 1*maxlen
        a = torch.exp(e)
        if self.mask:
            a = a * mask
        a = a / torch.sum(a, 1, keepdim=True)
        # ci = sum(aij*hj)
        # dim: 256*maxlen*maxlen*1-->256*1
        weighted_input = h * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class Attention_LSTM(nn.Module):
    def __init__(self, embedding_matrix, maxlen):
        super(Attention_LSTM, self).__init__()
        num_words = embedding_matrix.shape[0]
        # Dimension of embedding is num of columns in the matrix
        embedding_dim = embedding_matrix.shape[1]
        # Define an input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embedding_dim)
        # Embedding matrix actually is collection of parameter
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # Because we use pretrained embedding (GLove, Fastext,etc) so we turn off requires_grad-meaning we do not train gradient on embedding weight
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            embedding_dim,
            128,
            bidirectional=True,
            batch_first=True,
        )
        self.attention_layer = Attention(512, maxlen)
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        # pass input (tokens) through embedding layer
        x = self.embedding(x)
        # fit embedding to LSTM
        hidden, _ = self.lstm(x)
        # apply mean and max pooling on lstm output
        # attenrion layer over the hidden states (256*maxlen)
        attn = self.attention_layer(hidden)
        # fit out to self.out to conduct dimensionality reduction from 256 to 1
        out = self.out(attn)
        # return output
        return out


if __name__ == '__main__':
    x = torch.randn((16, 100))
    att_layer = Attention(100, 128)
    print(att_layer(x))


