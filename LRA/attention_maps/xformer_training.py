
from psf import PSFNet
from training_config import config
from psf_utils import DatasetCreator, count_params, seed_everything, TrainModel
from xformers import *
import sys
import torch
from torch import nn, optim
import torch_geometric

n_vec = 1024

seed_everything(42)

# Parse config
cfg_model = config['pathfinder']['models']['Linformer']
cfg_training = config['pathfinder']['training']
print(cfg_model)
print(cfg_training)
# Setting device
print(torch.cuda.get_device_name(cfg_training["device_id"]))
torch.cuda.set_device(cfg_training["device_id"])


if cfg_model["name"] == "transformer":
    net = TransformerHead(
        cfg_model["vocab_size"],
        cfg_model["dim"],
        cfg_model["heads"],
        cfg_model["depth"],
        n_vec,
        cfg_model["n_class"],
        cfg_model["problem"]
    )
elif cfg_model["name"] == "linformer":
    net = LinformerHead(
        cfg_model["vocab_size"],
        cfg_model["dim"],
        cfg_model["heads"],
        cfg_model["depth"],
        n_vec,
        cfg_model["n_class"],
        cfg_model["problem"]
    )
elif cfg_model["name"] == "performer":
    net = PerformerHead(
        cfg_model["vocab_size"],
        cfg_model["dim"],
        cfg_model["heads"],
        cfg_model["depth"],
        n_vec,
        cfg_model["n_class"],
        cfg_model["problem"]
    )
print('Number of the trainable parameters', count_params(net))
# print(net)
# sys.exit()

# Read the data
data = torch.load('pathfinder32_all_train.pt')
labels = torch.load('pathfinder32_all_train_targets.pt').to(torch.int64)

data_test = torch.load('pathfinder32_all_test.pt')
labels_test = torch.load('pathfinder32_all_test_targets.pt').to(torch.int64)

data_val = torch.load('pathfinder32_all_val.pt')
labels_val = torch.load('pathfinder32_all_val_targets.pt').to(torch.int64)

loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(
        net.parameters(),
        lr=cfg_training['learning_rate']
    )

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
    num_workers=4
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
    num_workers=4
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
    num_workers=4
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
    model_name=cfg_model["name"],
    saving_criteria=75
)
