import torch
from torch import nn, optim

from pathfinderX_utils_aug import SMFNet, DatasetCreator, count_params, seed_everything
from torch.utils.data import Dataset
import sys
import torch_geometric
from tqdm import tqdm
from datetime import datetime
from torchvision import datasets, transforms
import os 
from PIL import Image, ImageOps
from torch_sparse import spmm, spspmm
import tensorflow as tf
import pandas as pd
from torchvision import datasets, transforms


seed_everything(42)

# print(all_imgs)
# sys.exit()

# Setting device
device_id = 0
print(torch.cuda.get_device_name(device_id))

torch.cuda.set_device(device_id) 

cfg = {
    'batch_size': 20,
    'vocab_size': 225,
    'embedding_size': 32,
    'n_vec': 1024,
    'n_W': 10,
    'Ws': [32, 'GELU'],
    'n_links': 11,
    'Vs': [32, 'GELU'],
    'n_channels_V': 32,
    'n_class': 2,
    'use_dropout': False,
    'use_cuda': True,
    'residuals': False,
}

# Initialize the SMF net
net = SMFNet(cfg)
print('Number of the trainable parameters', count_params(net))

class ChangedSMF(SMFNet):
        def forward(self, data):

            data = self.embedding(data) + self.pos_embedding(torch.arange(0, self.n_vec).expand(self.batch_size, self.n_vec).cuda())
            V = self.gs(data)

            # print('V size:', V.size())
            if self.use_dropout:
                V = self.dropout1(V)
            if self.residuals:
                res_conn = V

            W_final = torch.eye(self.n_vec, self.n_vec).cuda()
            for i in range(self.n_W):
                W = self.fs[i](data)

                V = spmm(
                    self.chord_indicies,
                    W.reshape(W.size(0), W.size(1) * W.size(2)), 
                    self.n_vec,
                    self.n_vec,
                    V
                )

                W_final = spmm(
                    self.chord_indicies,
                    W.reshape(W.size(0), W.size(1) * W.size(2)), 
                    self.n_vec,
                    self.n_vec,
                    W_final
                )
                
            if self.residuals:
                V = V + res_conn

            if self.use_dropout:
                V = self.dropout2(V)

            V = self.final(V.view(V.size(0), -1))
            return V, W_final

net = ChangedSMF(cfg)
net.load_state_dict(torch.load('pathfinder_epoch9_acc79.03548559231591_bs32_es32_cV32_dropFalse.pt'))
net.eval()


data_test = torch.load('pathfinder32_all_test.pt')
labels_test = torch.load('pathfinder32_all_test_targets.pt').to(torch.int64)

if cfg['use_cuda']:
    net = net.cuda()

df = pd.read_csv('img_paths.csv')
# print(df.iloc[14][0])
# sys.exit()
class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, data, labels, img_paths):
        self.data = data
        self.labels = labels
        self.img_paths = img_paths

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.data[index]
        Y = self.labels[index].to(dtype=torch.long)
        # Y = self.labels[index]
        return (X, Y)

    def __len__(self):
        return len(self.labels)


# Prepare the testing loader
testset = DatasetCreator(
    data = data_test,
    labels = labels_test,
    img_paths = df
)

testloader = torch_geometric.data.DataLoader(
    testset,
    batch_size=cfg['batch_size'],
    shuffle=False,
    drop_last=False,
    num_workers=1
)


import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure

def take_ind_around(central_ind):
    central_ind = int(central_ind)
    up = [central_ind - 1 - 32, central_ind - 32, central_ind + 1 - 32]
    middle = [central_ind - 1, central_ind, central_ind + 1]
    down = [central_ind - 1 + 32, central_ind + 32, central_ind + 1 + 32]
    return up+middle+down

print(len(df))
# sys.exit()
correct = 0
total = 0
val_loss = 0.0
predictions = []
ground_truth = []

for i, (X, Y) in tqdm(enumerate(testloader), total=len(testloader)):
    X = X.cuda()
    Y = Y.cuda()
    pred, att_map = net(X)
    # print(att_map.size())
    
    _, predicted = pred.max(1)
    predictions.extend(predicted.tolist())
    ground_truth.extend(Y.tolist())
    total += Y.size(0)
    correct += predicted.eq(Y).sum().item()

    # att map visualization
    # tran = transforms.Compose([transforms.Normalize(0.5, 0.5), transforms.ToPILImage()])
    for im_index in range(cfg['batch_size']):
        # print(torch.matrix_rank(att_map[im_index]))
        # sys.exit()
        img_path = df.iloc[i * cfg['batch_size'] + im_index][0]
        fold, name = img_path.split('/')[-2:]
        # print(fold, name)
        if str(fold) == '107' and name == 'sample_818.png':
            print('saving pdf')
            sorted, indices = torch.topk(X[im_index], 2)
            indices_full = take_ind_around(indices[0]) + take_ind_around(indices[1])
            ddf = att_map[im_index].cpu().T.reshape((1024, 32, 32))[indices_full].sum(0)
            ddf = ddf.detach().numpy()
            ddf = ddf - ddf.min()
            ddf = ddf.clip(np.quantile(ddf, 0.7), np.quantile(ddf, 1.0)) ** .5
            img = plt.imread(img_path)

            # figure, axs = plt.subplots(1, 2)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(f'att_matr_path/1{fold}_{name[:-4]}.pdf', bbox_inches='tight', pad_inches=0)
            # axs[0].set_title('Original image')
            plt.imshow(ddf, cmap=plt.get_cmap('inferno'))
            plt.axis('off')
            plt.colorbar()
            plt.savefig(f'att_matr_path/2{fold}_{name[:-4]}.pdf', bbox_inches='tight', pad_inches=0)
            # axs[1].set_title('Att Map')

print("Test accuracy: {}".format(100.*correct/total))
accuracy = int(100.*correct/total)
print(accuracy)
print('len pred', len(predictions))
print('len gt', len(ground_truth))
leng = len(predictions)
df['targets'] = ground_truth
df['predictions'] = predictions
df.to_csv('inference_p32.csv', index=False)
print(df)
