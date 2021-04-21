from SMF_torch_deep import *
from torch_test import *
# Load

cfg: Dict[str, List[int]] = {
    'folder_name': ['generate_permute_data_gaussian'],
    'f': [13, 10],
    'g': [13, 10],
    'n_layers': [4],
    'N': [16],
    'd': [5],
    'disable_masking': [False],
    'LR': [0.001],
    'optimizer': ['Adam'],
    'batch_size': [100],
    'n_epochs': 200
}


class ChangedSMF(SMFNet):

    def forward(self, X):
        V0 = self.g(X.float())
        print('V0', np.round(V0.detach().numpy(), 1))
        for m in range(len(self.fs)):
            if self.disable_masking:
                W = self.fs[m](X.float())
            else:
                W = self.fs[m](X.float()) * self.chord_mask
            print('Chord', np.round(self.chord_mask.detach().numpy(), 1))
            print(f'W{m}', np.round(W.detach().numpy(), 1))
            V0 = torch.matmul(W, V0)
        return V0


def Changed_SMF_full(cfg: Dict[str, List]) -> ChangedSMF:
    model = ChangedSMF(
        g=make_layers_g(cfg),
        fs=nn.ModuleList(
            [make_layers_f(cfg) for _ in range(cfg['n_layers'][0])]
        ),
        N=cfg['N'][0],
        disable_masking=cfg['disable_masking'][0]
    )
    return model

#Train
# one_experient(cfg)

model = Changed_SMF_full(cfg)
model.load_state_dict(torch.load('SparseFactorization/output/model/final_model.pth'))
model.eval()

X = np.genfromtxt('SparseFactorization/train/generate_permute_data_gaussian/X/X_0.csv', delimiter=',')
print('X', np.round(X, 1))
X_gt = model(torch.from_numpy(X)).detach().numpy()
print('X_gt', np.round(X_gt, 1))
Y = np.genfromtxt('SparseFactorization/train/generate_permute_data_gaussian/Y/Y_0.csv', delimiter=',')
print('Y', np.round(Y, 1))