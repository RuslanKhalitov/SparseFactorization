import os
import tqdm
import random
import numpy as np
import numbers
import sklearn


def seed_everything(seed=1234):
    """
    Fixes random seeds, to get reproducible results.

    :param seed: a random seed across all the used packages
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def swap_columns(X, src_cols, tgt_cols):
    temp_src = X[:, src_cols]
    temp_tgt = X[:, tgt_cols]
    X[:, src_cols] = temp_tgt
    X[:, tgt_cols] = temp_src


def generate_permute_data_gaussian(N, d, noise=None):
    """
    Generate a permuted data set from Gaussian distribution,
    where for most data instances, the first two columns are exchanged with the last two,
    while there is no permutation for some instances (specified by 'noise' argument)
    :param N: number of data instances
    :param d: dimensionality
    :param noise: out-of-pattern noise level
    :return: N x d data
    """
    X = np.random.randn(N, d)
    Y = X.copy()
    permuate_range = 2
    assert d > permuate_range, "d must be larger than %d" % permuate_range
    swap_columns(Y, np.arange(0, permuate_range), np.arange(d-permuate_range, d))
    if noise is not None:
        assert (isinstance(noise, numbers.Number) and 0 < noise < 1), "noise must be a number between 0 and 1"
        ind = np.random.choice(np.arange(N), round(N * noise))
        Y[ind, :] = X[ind, :]
    return X, Y


def generate_permute_data_iris(noise=None):
    data = sklearn.datasets.load_iris()
    X = data.data
    N = X.shape[0]
    Y = X.copy()
    Y[:, [0]] = Y[:, [3]]
    if noise is not None:
        assert (isinstance(noise, numbers.Number) and 0 < noise < 1), "noise must be a number between 0 and 1"
        ind = np.random.choice(np.arange(N), round(N * noise))
        Y[ind, :] = X[ind, :]
    return X, Y


def generate_permute_data_sine(N, d, sigma=0.1, noise=None):
    xrange = np.linspace(0, 15, d)
    xrange = np.tile(xrange, (N, 1))
    shift = np.random.rand(N, 1)
    scale = np.random.rand(N, 1) + 1.0
    X = np.sin(xrange/scale + shift)
    X += np.random.randn(*X.shape) * sigma
    Y = X.copy()
    permuate_range = 10
    assert d > permuate_range, "d must be larger than %d" % permuate_range
    swap_columns(Y, np.arange(0, permuate_range), np.arange(d-permuate_range, d))
    if noise is not None:
        assert (isinstance(noise, numbers.Number) and 0 < noise < 1), "noise must a number between 0 and 1"
        ind = np.random.choice(np.arange(N), round(N * noise))
        Y[ind, :] = X[ind, :]
    return X, Y


if __name__ == '__main__':
    seed_everything(1234)
    N_samples_test = 500
    N_samples_train = 1000
    N_all = N_samples_test + N_samples_train
    N = 10
    d = 3

    # Generate permute gaussian data
    dirName = 'permute_gaussian'
    try:
        os.makedirs('SparseFactorization/train/' + dirName)
        os.makedirs('SparseFactorization/test/' + dirName)
    except FileExistsError:
        print("Directories already exist")

    for i in range(N_all):
        X, Y = generate_permute_data_gaussian(N, d, noise=0.95)
        if i <= N_samples_train:
            np.savetxt(f"SparseFactorization/train/permute_gaussian/X_{i}.csv", X, delimiter=",")
            np.savetxt(f"SparseFactorization/train/permute_gaussian/Y_{i}.csv", Y, delimiter=",")
        else:
            np.savetxt(f"SparseFactorization/test/permute_gaussian/X_{i}.csv", X, delimiter=",")
            np.savetxt(f"SparseFactorization/test/permute_gaussian/Y_{i}.csv", Y, delimiter=",")

    # np.random.seed(0)
# X, Y = generate_permute_data_sine(100, 100, noise=0.5)
# import matplotlib.pyplot as plt
# i = 10
# plt.plot(X[i, :], 'r')
# plt.plot(Y[i, :], 'b')
# plt.show()
