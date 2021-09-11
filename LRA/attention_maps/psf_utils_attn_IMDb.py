import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

token_index = {'D': 0, 'V': 1, 'U': 2, 'A': 3, '<': 4, '7': 5, 'O': 6, 'p': 7, 'u': 8, 'N': 9, 'm': 10, 'c': 11, 'I': 12, 'k': 13, '0': 14, 'T': 15, 'R': 16, '4': 17, 'o': 18, 'x': 19, 'd': 20, '-': 21, 'Z': 22, 'X': 23, '`': 24, 'z': 25, '"': 26, 'a': 27, '~': 28, ')': 29, 'i': 30, 'f': 31, 'h': 32, 'B': 33, '!': 34, '*': 35, '$': 36, ':': 37, 'l': 38, '2': 39, 'Q': 40, '|': 41, '>': 42, '@': 43, 'b': 44, '#': 45, '3': 46, 'H': 47, '8': 48, 'Y': 49, 'q': 50, '9': 51, 'g': 52, '%': 53, '6': 54, '?': 55, ';': 56, 'v': 57, ' ': 58, '[': 59, '^': 60, 'E': 61, '_': 62, 's': 63, 'S': 64, '5': 65, 'F': 66, 't': 67, 'G': 68, 'n': 69, ']': 70, '/': 71, '}': 72, 'e': 73, 'y': 74, 'M': 75, '&': 76, 'K': 77, '1': 78, 'L': 79, '+': 80, "'": 81, 'J': 82, 'r': 83, 'j': 84, 'C': 85, '(': 86, '=': 87, 'W': 88, '.': 89, 'w': 90, ',': 91, '{': 92, '\\': 93, 'P': 94, '<PAD>': 95, 'CLS': 96}


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

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.data[index]
        Y = self.labels[index].to(dtype=torch.long)
        return (X, Y)

    def __len__(self):
        return len(self.labels)


def count_params(net):
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return n_params


def attention_visualization(W_final, text, labels, preds):
    sentences_last_batch = []
    reverse_word_map = {v: k for k, v in token_index.items()}
    for t in text:
        sentence = []
        for s in t:
            s = int(s.cpu().numpy())
            if reverse_word_map[s] != '<PAD>':
                sentence.append(reverse_word_map[s])
        sentences_last_batch.append(sentence)
    texts = sentences_last_batch
    W_final = torch.abs(W_final)
    W_final = torch.nn.functional.normalize(W_final, dim=0)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(100, 10.0), facecolor="w")
    axes.tick_params(axis='y', which='major', labelsize=0, length=0)
    axes.tick_params(axis='x', which='major', length=0)
    input_sentence = [x for x in texts[0]]
    axes.imshow(W_final[0, 0, :len(input_sentence)].repeat(10).view(-1, len(input_sentence)).detach().cpu().numpy(), cmap="Greys", extent=[0,len(input_sentence),0,20])
    axes.set_xticks(np.arange(len(input_sentence)))
    axes.set_xticklabels(input_sentence)
    print(labels[0])
    if int(int(labels[0].cpu().numpy())) == 1:
        s = "positive"
    else:
        s = "negative"
    print(f'label:{labels[0]}, preds:{int(preds[0])}')
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(axes.get_yticklabels(), visible=False)
    plt.savefig(f'./results/W_{s}.png', bbox_inches='tight')
