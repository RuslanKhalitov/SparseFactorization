
from psf import PSFNet
from psf_training_config import config
from psf_utils import DatasetCreator, count_params, seed_everything, attention_visualization

import sys
import torch
from torch import nn, optim
import torch_geometric
from tqdm import tqdm
from torch_sparse import spmm

seed_everything(42)

# Parse config
cfg_model = config['imdb']['model']
cfg_training = config['imdb']['training']


# Setting device
print(torch.cuda.get_device_name(cfg_training["device_id"]))
torch.cuda.set_device(cfg_training["device_id"])

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
    use_pos_embedding=cfg_model["use_pos_embedding"],
    problem=cfg_model["problem"]
)
print('Number of trainable parameters', count_params(net))
net.load_state_dict(torch.load('imdb_epoch138_acc78.24503841229193.pt'))
net.eval()

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(
        net.parameters(),
        lr=cfg_training['learning_rate']
    )

data_test = torch.load('IMDB_test.pt').to(torch.int64)
labels_test = torch.load('IMDB_test_targets.pt').to(torch.int64)

if cfg_model['pooling_type'] == 'CLS':
    cls_token_data_test = torch.tensor([[cfg_model['vocab_size'] - 1]*data_test.size(0)]).T
    data_test = torch.cat([cls_token_data_test, data_test], -1)


if cfg_model['use_cuda']:
    net = net.cuda()

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

correct = 0
total = 0
val_loss = 0.0
predictions = []
ground_truth = []
for i, (X, Y) in tqdm(enumerate(testloader), total=len(testloader)):
    X = X.cuda()
    Y = Y.cuda()
    pred, W_val = net(X)
    _, predicted = pred.max(1)
    total += Y.size(0)
    correct += predicted.eq(Y).sum().item()

    attention_visualization(W_val, X, labels=Y, preds=predicted)
    accuracy_val = 100. * correct / total


