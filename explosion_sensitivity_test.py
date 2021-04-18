# Importing the required packages
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
import time
from SMF_torch_deep import *
import matplotlib.pyplot as plt
from permute_data import *
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

LEARNING_RATE = 1e-5
LR_FACTOR = 0.33  # for Adam optimization
NUM_EPOCHS = 100

cfg: Dict[str, List[int]] = {
    'f': [16, 32, d],
    'g': [16, 32, d]
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


class MyPermuteData(Dataset):
    def __init__(self, D, N, d):
        """
        :param D: Length of the dataset
        """
        super(MyPermuteData, self).__init__()
        self.X = []*D
        self.Y = []*D
        for i in range(D):
            X, Y = generate_permute_data_sine(N, d)
            self.X.append(torch.tensor(X))
            self.Y.append(torch.tensor(Y))

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
    train_losses = []
    test_losses = []
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33, random_state=42)
    for i in range(NUM_EPOCHS):
        train_loss = 0.0
        val_loss = 0.0
        star_time = time.time()
        for j in range(len(X_train)):
            V0 = model(X_train[j])
            loss = criterion(Y_train[j].float(), V0)
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_losses.append(train_loss.detach().numpy())
        epoch_time = (time.time()-star_time)
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        norms.append(total_norm)

        with torch.no_grad():
            for k in range(len(X_val)):
                output = model(X_val[k])
                loss = criterion(output, Y_val[k].float())
                val_loss += loss
            test_losses.append(val_loss.detach().numpy())

        if (i+1)%10 == 0:
            val_info = str(
                f'epoch {i+1}\t'
                f'train_loss {train_loss:.4f}\t'
                f'val_loss {val_loss:.4f}\t'
                f'train_time {epoch_time:.4f}\t'
            )
            print(val_info)

    return train_losses, norms, test_losses


def show_results(losses, norms):
    fig = plt.gcf()
    fig.suptitle(f'Learning rate {LEARNING_RATE}', fontsize=18)
    plt.subplot(1, 2, 1)
    plt.xlabel("epochs", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(losses, 'b')
    plt.title("training losses", fontsize=16)
    plt.subplot(1, 2, 2)
    plt.plot(norms, 'b')
    plt.xlabel("epochs", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("validation losses", fontsize=16)
    #plt.show()
    plt.savefig(f'Learning rate {LEARNING_RATE}.png')

if __name__ == '__main__':
    seed_everything(1234)
    DS = MyDataset(500, 16, 2)
    #print(DS.X)
    Model = model = SMF_full(cfg)
    print(model)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses, norms, test_losses = train(DS.X, DS.Y, model, criterion, optimizer)
    show_results(train_losses, test_losses)
