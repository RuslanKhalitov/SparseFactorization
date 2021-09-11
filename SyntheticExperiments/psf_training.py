from psf import PSFNet
from synthetic_training_config import config
from psf_utils import DatasetCreator, count_params, seed_everything, TrainModel

import sys
import torch
from torch import nn, optim
import torch_geometric
import numpy as np

problem = "adding" # "adding" or "order"
n_vec = 16384      # from 2**7 to 2**14
print(int(np.log2(n_vec)))

# Feel free to change the random seed
seed_everything(42)

# Parse config
cfg_model = config[problem]['models']['PSF']
cfg_training = config[problem]['training']


# Setting device
print(torch.cuda.get_device_name(cfg_training["device_id"]))
torch.cuda.set_device(cfg_training["device_id"])


# Initialize PSFNet
net = PSFNet(
    vocab_size=cfg_model["vocab_size"],
    add_init_linear_layer = cfg_model["add_init_linear_layer"],
    embedding_size=cfg_model["dim"],
    n_vec=n_vec, # defined at start
    n_W=int(np.log2(n_vec)),
    Ws=cfg_model["Ws"],
    V=cfg_model["V"],
    n_channels_V=cfg_model["n_channels_V"],
    n_class=cfg_model["n_class"],
    pooling_type=cfg_model["pooling_type"],
    head=cfg_model["head"],
    use_cuda=cfg_model["use_cuda"],
    use_residuals=cfg_model["use_residuals"],
    use_pos_embedding=cfg_model["use_pos_embedding"],
    problem=cfg_model["problem"]
)
print('Number of trainable parameters', count_params(net))
# print(net)


optimizer = optim.Adam(
        net.parameters(),
        lr=cfg_training['learning_rate']
    )

if problem == 'adding':
    loss = nn.MSELoss()
elif problem == 'order':
    loss = nn.CrossEntropyLoss()

# Read the data
data = torch.load(f'{problem}_{n_vec}_train.pt')
labels = torch.load(f'{problem}_{n_vec}_train_target.pt')

data_val = torch.load(f'{problem}_{n_vec}_val.pt')
labels_val = torch.load(f'{problem}_{n_vec}_val_target.pt')

data_test = torch.load(f'{problem}_{n_vec}_test.pt')
labels_test = torch.load(f'{problem}_{n_vec}_test_target.pt')



if cfg_model['use_cuda']:
    net = net.cuda()

# Prepare the training loader
trainset = DatasetCreator(
    data = data,
    labels = labels
)

trainloader = torch_geometric.data.DataLoader(
    trainset,
    batch_size=cfg_training['batch_size'],
    shuffle=True,
    drop_last=True,
    num_workers=1
)

# Prepare the validation loader
valset = DatasetCreator(
    data = data_val,
    labels = labels_val
)

valloader = torch_geometric.data.DataLoader(
    valset,
    batch_size=cfg_training['batch_size'],
    shuffle=False,
    drop_last=True,
    num_workers=1
)

# Prepare the testing loader
testset = DatasetCreator(
    data = data_test,
    labels = labels_test
)

testloader = torch_geometric.data.DataLoader(
    testset,
    batch_size=cfg_training['batch_size'],
    shuffle=False,
    drop_last=True,
    num_workers=1
)


TrainModel(
    net=net,
    trainloader=trainloader,
    valloader=valloader,
    testloader=testloader,
    n_epochs=cfg_training['num_train_steps'],
    test_freq=cfg_training['eval_frequency'],
    optimizer=optimizer,
    loss=loss,
    problem=cfg_model['problem'],
    saving_criteria=99.5
)

