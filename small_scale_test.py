import torch
import random
import os
import numpy as np
from SMF_torch_deep import *
import matplotlib.pyplot as plt
cfg: Dict[str, List[int]] = {
    'D': [500],
    'n_layers': [1],
    'N': [3],
    'd': [2],
    'disable_masking': [True],
    'num_epoch': [100],
    'LR': [0.001]
    }


class ChangedSMF(SMFNet):

    def forward(self, X):
        V0 = X.float()
        W = torch.tensor((self.N, self.N))
        for m in range(len(self.fs)):
            if self.disable_masking:
                W = self.fs[m](X.float())
                #print(W)
            else:
                W = self.fs[m](X.float())
            V0 = torch.matmul(W, V0)
        return W, V0


def make_simple_f(H, d):
    return nn.Linear(d, H, bias=True)


def make_simple_g(N):
    return torch.eye(N)


def simple_SMF(cfg: Dict[str, List]):
    model = ChangedSMF(
        g=make_simple_g(cfg['N'][0]),
        fs=nn.ModuleList(
            [make_simple_f(cfg['N'][0], cfg['d'][0]) for _ in range(cfg['n_layers'][0])]
        ),
        N=cfg['N'][0],
        disable_masking=cfg['disable_masking'][0]
    )
    return model


def seed_everything(seed=1234):
    """
    Fixes random seeds, to get reproducible results.
    :param seed: a random seed across all the used packages
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_dataset(D, N, d, sigma):
    A = torch.rand((N, N))
    X = [] * D
    Y = [] * D
    for i in range(D):
        mu = np.mean(np.random.normal(0, 1, size=(N, d)))
        x = torch.normal(mean=mu, std=sigma**2, size=(N, d))
        y = torch.matmul(A, x)
        X.append(x)
        Y.append(y)

    return A, X, Y


def train(X, Y, model, criterion, optimizer):
    losses = []
    for i in range(cfg['num_epoch'][0]):
        train_loss = 0.0
        for j in range(len(X)):
            optimizer.zero_grad()
            W, V0 = model(X[j])
            loss = criterion(Y[j].float(), V0)
            loss.backward()
            optimizer.step()
            train_loss += loss
        losses.append(train_loss.detach().numpy())
        if (i+1) % 10 == 0:
            print(f'epoch{i+1}\t'
                  f'loss:{train_loss/len(X)}')
        if (i+1) == cfg['num_epoch'][0]:
            print(W)

    return losses


if __name__ == '__main__':
    seed_everything(1234)
    A, X, Y = generate_dataset(cfg['D'][0], cfg['N'][0], cfg['d'][0], 0.5)
    model = simple_SMF(cfg)
    print(model)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['LR'][0])
    losses = train(X, Y, model, criterion, optimizer)
    print(A)
    plt.plot(losses, 'b')
    plt.title("training losses", fontsize=14, pad=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

