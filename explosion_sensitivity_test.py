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
import pandas as pd
from Analysis import Analysis, PlotGraphs

LEARNING_RATE = 1e-7
LR_FACTOR = 0.33  # for Adam optimization
NUM_EPOCHS = 10

cfg: Dict[str, List[int]] = {
    'f': [16, 32],
    'g': [16, 32],
    'n_layers': [4],
    'N': [16],
    'd': [2]
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
    norms = []
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.33, random_state=42)
    for i in range(NUM_EPOCHS):
        train_loss = 0.0
        val_loss = 0.0
        star_time = time.time()
        optimizer.zero_grad()
        for j in range(len(X_train)):
            V0 = model(X_train[j])
            loss = criterion(Y_train[j].float(), V0)
            loss.backward()
            optimizer.step()
            train_loss += loss
        stats_container = Analysis(model, cfg).stats_on_params()
        final_dict["train_loss"].append(train_loss.detach().numpy())
        final_dict['grad_g_value'].append(stats_container["g_weights_grads"])
        final_dict['grad_g_bias'].append(stats_container["g_biases_grads"])
        final_dict['grad_f_value'].append(stats_container["fs_weights_grads"])
        final_dict['grad_f_bias'].append(stats_container["fs_weights_grads"])
        final_dict['g_weight_std'].append(stats_container["g_weights"][0])
        final_dict['g_weight_mean'].append(stats_container["g_weights"][1])
        final_dict['g_weight_max'].append(stats_container["g_weights"][2])
        final_dict['g_bias_std'].append(stats_container["g_biases"][0])
        final_dict['g_bias_mean'].append(stats_container["g_biases"][1])
        final_dict['g_bias_max'].append(stats_container["g_biases"][2])
        final_dict['fs_weight_std'].append(stats_container["fs_weights"][0])
        final_dict['fs_weight_mean'].append(stats_container["fs_weights"][1])
        final_dict['fs_weight_max'].append(stats_container["fs_weights"][2])
        final_dict['fs_bias_std'].append(stats_container["fs_biases"][0])
        final_dict['fs_bias_mean'].append(stats_container["fs_biases"][1])
        final_dict['fs_bias_max'].append(stats_container["fs_biases"][2])
        epoch_time = (time.time()-star_time)
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        norms.append(total_norm)

        save_weights(model, i+1, LEARNING_RATE)

        with torch.no_grad():
            for k in range(len(X_val)):
                output = model(X_val[k])
                loss = criterion(output, Y_val[k].float())
                val_loss += loss
            final_dict["test_loss"].append(val_loss.detach().numpy())

        if (i+1)%10 == 0:
            val_info = str(
                f'epoch {i+1}\t'
                f'train_loss {train_loss:.4f}\t'
                f'val_loss {val_loss:.4f}\t'
                f'train_time {epoch_time:.4f}\t'
            )
            print(val_info)

    return final_dict

def save_weights(model, epoch, LR):
    #parent_dir = "/home/dashuo/PycharmpProjects/SFM_torch/"
    #directory = 'LR'+str(LR)+'/weights/g'
    #path = os.path.join(parent_dir, directory)
    #os.mkdir(path)
    g_0_weight = model.g[0].weight
    x_df = pd.DataFrame(g_0_weight.detach().numpy())
    g_outfile_name = 'weights/g/' + 'g_0_epoch' + str(epoch)+'_weight'
    x_df.to_csv(g_outfile_name+'.csv', sep=',', index=False)
    g_2_weight = model.g[2].weight
    x_df = pd.DataFrame(g_2_weight.detach().numpy())
    g_outfile_name = 'weights/g/'+'g_2_epoch' + str(epoch)+'_weight'
    x_df.to_csv(g_outfile_name+'.csv', sep=',', index=False)
    idx = [0, 2]
    for i in range(4):
        for j in idx:
            weight = model.fs[i][j].weight
            x_df = pd.DataFrame(weight.detach().numpy())
            outfile_name = 'weights/f/f'+str(i)+'/f_'+str(i)+str(j)+'_epoch'+str(epoch)+'_weight'
            x_df.to_csv(outfile_name+'.csv', sep = ',', index = False)


def show_results(train_losses, val_losses, norms):
    fig = plt.gcf()
    fig.suptitle(f'Learning rate {LEARNING_RATE}', fontsize=14)
    fig.tight_layout()
    plt.subplot(1, 3, 1)
    plt.xlabel("epochs", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=6)
    plt.plot(train_losses, 'b')
    plt.title("training losses", fontsize=12, pad=12)


    plt.subplot(1, 3, 2)
    plt.xlabel("epochs", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=6)
    plt.plot(val_losses, 'b')
    plt.title("validation losses", fontsize=12, pad=12)

    plt.subplot(1, 3, 3)
    plt.xlabel("epochs", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=6)
    plt.plot(norms, 'b')
    plt.title("L2 norms of gradients", fontsize=12,  pad=12)

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
    final_dict = train(DS.X, DS.Y, model, criterion, optimizer)
    #show_results(train_losses, val_losses,  norms)
    p = PlotGraphs(final_dict)
    p.plot()
