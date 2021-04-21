# Importing the required packages
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
import time
import multiprocessing
from typing import List, Dict

# Importing local supplementary files
from SMF_torch_deep import *
from Analysis import Analysis, PlotGraphs

# Globals
YOUR_DIRECTORY_NAME = '/Users/ruslanhalitov/PycharmProjects'  # !!! CHANGE IT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 50
LEARNING_RATE = 1e-3
LR_STEP = 1
LR_FACTOR = 0.33  # for Adam optimization
NUM_WORKERS = multiprocessing.cpu_count()  # for parallel inference
NUM_EPOCHS = 100
NUM_ITERATIONS = 1000
INFERENCE_FREQUENCY = 10000
LOGGING_FREQUENCY = 20000
VAL_LOGGING_FREQUENCY = 20000
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


def load_data(batch_size, folder_name):
    """
    Prepares the training and testing dataloaders
    :return: train dataloader, test_dataloader
    """
    train_dataset = DatasetCreator(mode='train',
                                   X_folder=f'SparseFactorization/train/{folder_name}/X',
                                   Y_folder=f'SparseFactorization/train/{folder_name}/Y',
                                   )

    test_dataset  = DatasetCreator(mode='test',
                                   X_folder=f'SparseFactorization/train/{folder_name}/X',
                                   Y_folder=f'SparseFactorization/train/{folder_name}/Y',
                                   )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False)

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
    losses_ev = AverageMeter()
    best_val_score = 10**5
    end = time.time()

    model.train()
    for i, (X, X_gt) in enumerate(train_loader):
        # !!!
        optimizer.zero_grad()

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
        # if i % INFERENCE_FREQUENCY == INFERENCE_FREQUENCY - 1:
        #     print('Model evaluation')
        #     print('-' * 30)
        #
        #     losses_ev = AverageMeter()
        #     num_steps_eval = len(test_loader)
        #     model.eval()
        #
        #     with torch.no_grad():
        #         for j, (X, X_gt) in enumerate(test_loader):
        #             output = model(X)
        #             loss_ev = criterion(output, X_gt.float())
        #             losses_ev.update(loss_ev.data.item(), X.size(0))
        #
        #             if j % VAL_LOGGING_FREQUENCY == 0:
        #                 val_info = str(
        #                     f'epoch {epoch} [{j}/{num_steps_eval}]\t'
        #                     f'loss {losses_ev.val:.4f} ({losses_ev.avg:.4f})\t'
        #                 )
        #                 print(val_info)

        # Saving model if it is better than the previous best
        # if float(losses_ev.avg) < best_val_score:
        #     print('Saving the model')
        #     print(f'Previous val score {best_val_score}, current val score {float(losses_ev.avg)}')
        #
        #     torch.save(model.state_dict(), f'best_model_val_{epoch}.pth')
        #     best_val_score = losses_ev.avg


        # model.train()
        end = time.time()

    # Loss for validation
    model.eval()

    with torch.no_grad():
        for j, (X, X_gt) in enumerate(test_loader):
            output = model(X)
            loss_ev = criterion(output, X_gt.float())
            losses_ev.update(loss_ev.data.item(), X.size(0))

        return round(losses.avg, 1), round(losses_ev.avg, 1)


def one_experient(cfg):

    seed_everything(1234)
    assert str(os.getcwd()) == YOUR_DIRECTORY_NAME,\
        "Please specify parameter YOUR_DIRECTORY_NAME"

    model = SMF_full(cfg)
    # print(model)

    criterion = torch.nn.MSELoss(reduction='sum')

    # Mapping optimizer
    optim_mapped = {
        'Adam': torch.optim.Adam,
        'RMSP': torch.optim.RMSprop,
        'SGD': torch.optim.SGD
    }[cfg['optimizer'][0]]

    optimizer = optim_mapped(model.parameters(), lr=cfg['LR'][0])
    train_loader, test_loader = load_data(cfg['batch_size'][0], cfg['folder_name'][0])

    final_dict = {
        'train_loss': [],
        'test_loss': [],

        'grad_g_value': [],
        'grad_g_bias': [],
        'grad_f_value': [],
        'grad_f_bias': [],

        'g_weight_std': [],
        'g_weight_mean': [],
        'g_weight_max': [],

        'g_bias_std': [],
        'g_bias_mean': [],
        'g_bias_max': [],

        'fs_weight_std': [],
        'fs_weight_mean': [],
        'fs_weight_max': [],

        'fs_bias_std': [],
        'fs_bias_mean': [],
        'fs_bias_max': [],
    }

    for epoch in range(1, cfg['n_epochs'] + 1):

        loss, loss_ev = train(model, epoch, train_loader, test_loader, criterion, optimizer)
        final_dict['train_loss'].append(loss)
        final_dict['test_loss'].append(loss_ev)

        final_dict = Analysis(model, cfg, final_dict).stats_on_params()

        torch.save(model.state_dict(), "final_model_{}.pth".format(epoch))

    PlotGraphs(final_dict, cfg).plot()


if __name__ == '__main__':
    for LR in [0.01, 0.001, 0.0001]:
        for disable_mask in [False, True]:
            for optimizer in ['Adam', 'RMSP', 'SGD']:
                for batch_size in [100]:

                    cfg: Dict[str, List[int]] = {
                        'folder_name': ['generate_exp_data'],
                        'f': [13, 10],
                        'g': [13, 10],
                        'n_layers': [4],
                        'N': [16],
                        'd': [5],
                        'disable_masking': [disable_mask],
                        'LR': [LR],
                        'optimizer': [optimizer],
                        'batch_size': [batch_size],
                        'n_epochs': 200
                    }

                    one_experient(cfg)