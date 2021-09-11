from psf import PSFNet
from psf_training_config import config
from psf_utils import DatasetCreator, count_params, seed_everything, TrainPSF
import torch
from torch import nn, optim
from torch.utils.data import Dataset
import torch_geometric
from tqdm import tqdm
from torch_sparse import spmm, spspmm
import tensorflow as tf
import pandas as pd

seed_everything(42)

# Parse config
cfg_model = config['pathfinder']['model']
cfg_training = config['pathfinder']['training']


# Setting device
print(torch.cuda.get_device_name(cfg_training["device_id"]))
# torch.cuda.set_device(cfg_training["device_id"])
torch.cuda.set_device(0)

# Make an instance for extracting the attention map
class ChangedPSF(PSFNet):
    def forward(self, data):
        data = self.embedding(data)
        if self.use_pos_embedding:
            positions = torch.arange(0, self.n_vec).expand(data.size(0), self.n_vec)
            if self.use_cuda:
                positions = positions.cuda()
            pos_embed = self.pos_embedding(positions)
            data = data + pos_embed

        data = self.dropout1(data)
        V = self.g(data)
        V = self.dropout2(V)
        if self.use_residuals:
            res_conn = V
        
        # To extract the attention map
        W_final = torch.eye(self.n_vec, self.n_vec).cuda()

        for m in range(self.n_W):
            W = self.fs[m](data)
            V = spmm(
                self.chord_indicies,
                W.reshape(W.size(0), W.size(1) * W.size(2)), 
                self.n_vec,
                self.n_vec,
                V
            )
            # Construct the dense attention map
            W_final = spmm(
                    self.chord_indicies,
                    W.reshape(W.size(0), W.size(1) * W.size(2)), 
                    self.n_vec,
                    self.n_vec,
                    W_final
                )

            if self.use_residuals:
                V = V + res_conn

        V = self.dropout3(V)
        if self.pooling_type == 'CLS':
            V = V[:, 0, :]

        V = self.final(V.view(V.size(0), -1))
        return V, W_final

# Initialize ChangedPSFNet
net = ChangedPSF(
    vocab_size=cfg_model["vocab_size"],
    embedding_size=cfg_model["embedding_size"],
    n_vec=cfg_model["n_vec"],
    n_W=cfg_model["n_W"],
    Ws=cfg_model["Ws"],
    V=cfg_model["V"],
    n_channels_V=cfg_model["n_channels_V"],
    n_class=cfg_model["n_class"],
    pooling_type=cfg_model["pooling_type"],
    head=cfg_model["head"],
    use_cuda=cfg_model["use_cuda"],
    use_residuals=cfg_model["use_residuals"],
    dropout1_p=cfg_model["dropout1_p"],
    dropout2_p=cfg_model["dropout2_p"],
    dropout3_p=cfg_model["dropout3_p"],
    init_embedding_weights = cfg_model["init_embedding_weights"],
    use_pos_embedding=cfg_model["use_pos_embedding"],
    problem=cfg_model["problem"]
)
print('Number of trainable parameters', count_params(net))
net.load_state_dict(torch.load('pathfinder_epoch20_acc80.60120240480961.pt'))
net.eval()


data_test = torch.load('pathfinder32_all_test.pt')
labels_test = torch.load('pathfinder32_all_test_targets.pt').to(torch.int64)

if cfg_model['use_cuda']:
    net = net.cuda()

df = pd.read_csv('img_paths.csv')
# print(df.iloc[14][0])
# sys.exit()


# Prepare the testing loader
testset = DatasetCreator(
    data = data_test,
    labels = labels_test
)

testloader = torch_geometric.data.DataLoader(
    testset,
    batch_size=8,
    shuffle=False,
    drop_last=True,
    num_workers=2
)


import matplotlib.pyplot as plt
import numpy as np
import os
if not os.path.exists('att_matr_path'):
    os.makedirs('att_matr_path')

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
    for im_index in range(8):
        img_path = df.iloc[i * 8 + im_index][0]
        fold, name = img_path.split('/')[-2:]
        sorted, indices = torch.topk(X[im_index], 2)
        indices_full = take_ind_around(indices[0]) + take_ind_around(indices[1])
        ddf = att_map[im_index].cpu().T.reshape((1024, 32, 32))[indices_full].sum(0)
        ddf = ddf.detach().numpy()
        ddf = ddf - ddf.min()
        ddf = ddf.clip(np.quantile(ddf, 0.7), np.quantile(ddf, 1.0)) ** .5
        img = plt.imread(img_path)

        figure, axs = plt.subplots(1, 2, gridspec_kw = {'wspace':0.05, 'hspace':0.05})
        axs[0].imshow(img, cmap='gray')
        axs[0].axis('off')
        axs[1].imshow(ddf, cmap=plt.get_cmap('inferno'))
        axs[1].axis('off')
        plt.savefig(f'att_matr_path/{fold}_{name[:-4]}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

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
