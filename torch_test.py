# Importing the required packages
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
import time
import multiprocessing
# from permute_data import *

# Importing local supplementary files
from SMF_torch_extension import SMFNet

# Globals
YOUR_DIRECTORY_NAME = '/Users/ruslanhalitov/PycharmProjects'  # !!! CHANGE IT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
LR_STEP = 1
LR_FACTOR = 0.33  # for Adam optimization
NUM_WORKERS = multiprocessing.cpu_count()  # for parallel inference
NUM_EPOCHS = 10
NUM_ITERATIONS = 1000
INFERENCE_FREQUENCY = 300
LOGGING_FREQUENCY = 100
VAL_LOGGING_FREQUENCY = 30
MAX_STEPS_PER_EPOCH = 10**5


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


class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, mode, X_folder, Y_folder):
        print(f'Creating data loader - {mode}')
        assert mode in ['train', 'test']
        self.mode = mode
        self.X_folder = X_folder
        self.Y_folder = Y_folder
        self.list_of_X = os.listdir(self.X_folder)
        self.list_of_Y = os.listdir(self.Y_folder)
        assert len(self.list_of_X) == len(self.list_of_Y),\
            "The number of samples doesn't match the number of labels"

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        filename_X = self.list_of_X[index]
        filename_Y = self.list_of_Y[index]
        X = np.genfromtxt(f'{self.X_folder}/{filename_X}', delimiter=',')
        Y = np.genfromtxt(f'{self.Y_folder}/{filename_Y}', delimiter=',')
        return X, Y

    def __len__(self):
        return len(self.list_of_X)


def load_data():
    """
    Prepares the training and testing dataloaders
    :return: train dataloader, test_dataloader
    """
    train_dataset = DatasetCreator(mode='train',
                                   X_folder='SparseFactorization/train/permute_gaussian/X',
                                   Y_folder='SparseFactorization/train/permute_gaussian/Y',
                                   )

    test_dataset  = DatasetCreator(mode='test',
                                   X_folder='SparseFactorization/train/permute_gaussian/X',
                                   Y_folder='SparseFactorization/train/permute_gaussian/Y',
                                   )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    return train_loader, test_loader


class AverageMeter:
    """
    Stores the statistics on training time
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, epoch, train_loader, test_loader, criterion, optimizer):
    """
    The main training procedure.
    :param model: NN
    :param epoch: epoch number
    :param train_loader: train_loader
    :param test_loader: test_loader
    :param criterion: loss function
    :param optimizer: optimizer
    :return: None
    """
    num_steps = min(len(train_loader), MAX_STEPS_PER_EPOCH)
    batch_time = AverageMeter()
    losses = AverageMeter()
    best_val_score = 10**5
    end = time.time()

    model.train()
    for i, (X, X_gt) in enumerate(train_loader):
        V0 = model(X)
        loss = criterion(X_gt.float(), V0)
        losses.update(loss.data.item(), X.size(0))

        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)

        # Logging
        if i % LOGGING_FREQUENCY == 0:
            train_info = str(
                f'epoch {epoch} [{i}/{num_steps}]\t'
                f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
            )
            print(train_info)

        # Validation
        if i % INFERENCE_FREQUENCY == 0:
            print('Model evaluation')
            print('-' * 30)

            losses_ev = AverageMeter()
            num_steps_eval = len(test_loader)
            model.eval()

            with torch.no_grad():
                for j, (X, X_gt) in enumerate(test_loader):
                    output = model(X)
                    loss = criterion(output, X_gt.float())
                    losses_ev.update(loss.data.item(), X.size(0))

                    if j % VAL_LOGGING_FREQUENCY == 0:
                        val_info = str(
                            f'epoch {epoch} [{j}/{num_steps_eval}]\t'
                            f'loss {losses_ev.val:.4f} ({losses_ev.avg:.4f})\t'
                        )
                        print(val_info)

        # Saving model if it is better than the previous best
        if float(losses_ev.avg) < best_val_score:
            print('Saving the model')
            print(f'Previous val score {best_val_score}, current val score {float(losses_ev.avg)}')

            torch.save(model.state_dict(), f'best_model_val_{epoch}.pth')
            best_val_score = losses_ev.avg

        model.train()
        end = time.time()


if __name__ == '__main__':
    seed_everything(1234)
    assert str(os.getcwd()) == YOUR_DIRECTORY_NAME,\
        "Please specify parameter YOUR_DIRECTORY_NAME"

    D_in, H1, H2, n_layer = 5, 5, 16, 4
    model = SMFNet(D_in, H1, H2, n_layer)

    # for param in model.parameters():
    #     param.requires_grad = True

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, test_loader = load_data()

    for epoch in range(1, NUM_EPOCHS + 1):
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS))
        print('-' * 30)

        train(model, epoch, train_loader, test_loader, criterion, optimizer)

        torch.save(model.state_dict(), "final_model_{}.pth".format(epoch))

    # import matplotlib.pyplot as plt
    # i = 10
    # plt.plot(X_gt[i, :], 'r')
    # plt.plot(V0[i, :], 'b')
    # plt.show()
