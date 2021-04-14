# Importing the required packages
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
import time
from SMF_torch_deep import *
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-5
LR_FACTOR = 0.33  # for Adam optimization
NUM_EPOCHS = 100

cfg: Dict[str, List[int]] = {
    'f': [16, N],
    'g': [16, d]
}


class MyDataset(Dataset):
    def __init__(self, D, N, d):
        """
        :param D: Length of the dataset
        """
        super(MyDataset, self).__init__()
        self.X = []*D
        self.Y = []*D
        for i in range(D):
            self.X.append(torch.rand(size=(N, d)))
            self.Y.append(torch.rand(size=(N, d)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[index], self.Y[index]


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


def train(X, Y, model, criterion, optimizer):
    norms = []
    losses = []
    for i in range(NUM_EPOCHS):
        epoch_loss = 0.0
        star_time = time.time()
        for j in range(len(X)):
            V0 = model(X[j])
            loss = criterion(Y[j].float(), V0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        losses.append(epoch_loss.detach().numpy())
        epoch_time = (time.time()-star_time)
        if (i+1)%10 == 0:
            val_info = str(
                f'epoch {i+1}\t'
                f'loss {epoch_loss:.4f}\t'
                f'time {epoch_time:.4f}\t'
            )
            print(val_info)

        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        norms.append(total_norm)

    return losses, norms


def show_results(losses, norms):
    fig = plt.gcf()
    fig.suptitle(f'Learning rate {LEARNING_RATE}', fontsize=18)
    plt.subplot(1, 2, 1)
    plt.xlabel("epochs", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(losses, 'b')
    plt.title("losses change", fontsize=16)
    plt.subplot(1, 2, 2)
    plt.plot(norms, 'b')
    plt.xlabel("epochs", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("2 norms of gradients", fontsize=16)
    #plt.show()
    plt.savefig(f'Learning rate {LEARNING_RATE}.png')

if __name__ == '__main__':
    seed_everything(1234)
    DS = MyDataset(100, 16, 2)
    #print(DS.X)
    Model = model = SMF_full(cfg)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses, norms = train(DS.X, DS.Y, model, criterion, optimizer)
    show_results(losses, norms)
