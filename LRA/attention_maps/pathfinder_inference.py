from psf import PSFNet
from training_config import config
from psf_utils import DatasetCreator, count_params, seed_everything, TrainPSF
from xformers import TransformerHead, PerformerHead, LinformerHead

import torch
import torch_geometric
from tqdm import tqdm
from torch_sparse import spmm, spspmm
import tensorflow as tf
import pandas as pd
import sys 
import matplotlib.pyplot as plt
import numpy as np
import os

seed_everything(42)

# Parse config
backbone = 'Linformer' # PSF, Transformer, Linformer, Performer
cfg_model = config['pathfinder']['models'][backbone]
cfg_training = config['pathfinder']['training']

# Setting device
print(torch.cuda.get_device_name(cfg_training["device_id"]))
torch.cuda.set_device(cfg_training["device_id"])


# Make a class for extracting the attention map
class ChangedPSF(PSFNet):
    def forward(self, data):
        # Get embedding
        data = self.embedding(data)

        # Get positional embedding if needed
        if self.use_pos_embedding:
            positions = torch.arange(0, self.n_vec).expand(data.size(0), self.n_vec)
            if self.use_cuda:
                positions = positions.cuda()
            pos_embed = self.pos_embedding(positions)
            data = data + pos_embed

        # Apply the first dropout
        data = self.dropout1(data)

        # Get V 
        V = self.g(data)

        # Apply the second dropout
        V = self.dropout2(V)

        # Init residual connection if needed
        if self.use_residuals:
            res_conn = V

        # To extract the attention map
        W_final = torch.eye(self.n_vec, self.n_vec).cuda()

        # Iterate over all W
        for m in range(self.n_W):

            # Get W_m 
            W = self.fs[m](data)

            # Multiply W_m and V, get new V
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
            
            # Apply residual connection
            if self.use_residuals:
                V = V + res_conn

        # Apply the third dropout
        V = self.dropout3(V)
            
        if self.pooling_type == 'CLS':
            V = V[:, 0, :]

        V = self.final(V.view(V.size(0), -1))
        return V, W_final



# Changed to save Att Maps
if backbone == 'Performer':
    from performer_pytorch.performer_pytorch import *

    def changed_forward_performer(self, x, pos_emb = None, context = None, mask = None, context_mask = None, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))
        # get attention
        with torch.no_grad():
            qqq = q.squeeze(1)
            kkk = k.squeeze(1).view(k.size(0), k.size(-1), k.size(-2))
            A = torch.bmm(qqq, kkk)
            torch.save(A, 'attention.pt')
        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)

    Attention.forward = changed_forward_performer
elif backbone == 'Linformer':
    from linformer.linformer import *

    def changed_forward_linformer(self, x, context = None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # Get attention
        with torch.no_grad():
            qqq = queries
            kkk = keys.view(keys.size(0), keys.size(-1), keys.size(-2))
            A = torch.bmm(qqq, kkk)
            torch.save(A, 'attention.pt')

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

    LinformerSelfAttention.forward = changed_forward_linformer
elif backbone == 'Transformer':
    from torch.nn.modules.transformer import *

    def new_sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, A = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        torch.save(A, 'attention.pt')
        return self.dropout1(x)

    # import inspect
    # print(inspect.getsource(TransformerEncoderLayer._sa_block))
    TransformerEncoderLayer._sa_block = new_sa_block


# Initialize Network
if backbone == 'PSF':
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
    net.load_state_dict(torch.load('PSF_5.pt'))
elif backbone == 'Transformer':
    net = TransformerHead(
        cfg_model["vocab_size"],
        cfg_model["dim"],
        cfg_model["heads"],
        cfg_model["depth"],
        1024,
        cfg_model["n_class"],
        cfg_model["problem"]
    )
    net.load_state_dict(torch.load('transformer_6.pt'))
elif backbone == 'Linformer':
    net = LinformerHead(
        cfg_model["vocab_size"],
        cfg_model["dim"],
        cfg_model["heads"],
        cfg_model["depth"],
        1024,
        cfg_model["n_class"],
        cfg_model["problem"]
    )
    net.load_state_dict(torch.load('linformer_8.pt'))
elif backbone == 'Performer':
    net = PerformerHead(
        cfg_model["vocab_size"],
        cfg_model["dim"],
        cfg_model["heads"],
        cfg_model["depth"],
        1024,
        cfg_model["n_class"],
        cfg_model["problem"]
    )
    net.load_state_dict(torch.load('performer_2.pt'))

print('Number of trainable parameters', count_params(net))
net.eval()

# Read the testing data for inference
data_test = torch.load('pathfinder32_all_test.pt')
labels_test = torch.load('pathfinder32_all_test_targets.pt').to(torch.int64)

if cfg_model['use_cuda']:
    net = net.cuda()

# Paths dataframe
df = pd.read_csv('img_paths.csv')

# Prepare the testing loader
inference_batch_size = 8

testset = DatasetCreator(
    data = data_test,
    labels = labels_test
)

testloader = torch_geometric.data.DataLoader(
    testset,
    batch_size=inference_batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=2
)


def take_ind_around(central_ind):
    central_ind = int(central_ind)
    up = [central_ind - 1 - 32, central_ind - 32, central_ind + 1 - 32]
    middle = [central_ind - 1, central_ind, central_ind + 1]
    down = [central_ind - 1 + 32, central_ind + 32, central_ind + 1 + 32]
    return up+middle+down


def vis_attention_map(
    batch_size,
    batch_index,
    backbone='PSF',
    save_path='att_matr_path',
    paired=False,
    q_up=1,
    q_down=0.7):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for im_index in range(batch_size):
        img_path = df.iloc[batch_index * batch_size + im_index][0]
        fold, name = img_path.split('/')[-2:]
        sorted, indices = torch.topk(X[im_index], 2)
        indices_full = take_ind_around(indices[0]) + take_ind_around(indices[1])
        ddf = att_map[im_index].cpu().T.reshape((1024, 32, 32))[indices_full].mean(0)
        # else:
        #     ddf = att_map[im_index].cpu().T.reshape((32, 32))
        ddf = ddf.detach().numpy()
        ddf = ddf - ddf.min()
        ddf = ddf.clip(np.quantile(ddf, q_down), np.quantile(ddf, q_up)) ** .5
        img = plt.imread(img_path)

        if paired:
            figure, axs = plt.subplots(1, 2, gridspec_kw = {'wspace':0.05, 'hspace':0.05})
            axs[0].imshow(img, cmap='gray')
            axs[0].axis('off')
            axs[1].imshow(ddf, cmap=plt.get_cmap('inferno'))
            axs[1].axis('off')
            plt.show()
            plt.savefig(f'att_matr_path/{fold}_{backbone}_{name[:-4]}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            figure, axs = plt.subplots(1, 1)
            axs.imshow(ddf, cmap=plt.get_cmap('inferno'))
            axs.axis('off')
            plt.show()
            plt.savefig(f'att_map_final/{fold}_{backbone}_{name[:-4]}.png', bbox_inches='tight', pad_inches=0)
            plt.close()


correct = 0
total = 0
val_loss = 0.0
predictions = []
ground_truth = []

for batch_idx, (X, Y) in tqdm(enumerate(testloader), total=len(testloader)):
    X = X.cuda()
    Y = Y.cuda()
    if backbone=='PSF':
        pred, att_map = net(X)
    else:
        pred = net(X)
        att_map = torch.load('attention.pt')
    _, predicted = pred.max(1)
    predictions.extend(predicted.tolist())
    ground_truth.extend(Y.tolist())
    total += Y.size(0)
    correct += predicted.eq(Y).sum().item()

    # visualization for batch
    vis_attention_map(
        batch_size=inference_batch_size,
        batch_index=batch_idx,
        backbone=backbone,
        save_path='att_matr_path',
        paired=False,
        q_up=1,
        q_down=0.7
    )

