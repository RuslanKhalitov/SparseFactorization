"""
Generate a synthetic sequences for classification, where
1) the signal positions are distant
2) the signal positions can be slightly variable (controlled by the 'noise' :parameter
3) the class indicating signals follow XOR distribution;
4) that is, the classification cannot be done with just one signal position
"""
import numpy as np

# Background alphabet
bk_alphabet = np.array(['a', 'b', 'c', 'd'])
bk_alphabet_size = len(bk_alphabet)

# class indicating alphabet
cls_alphabet = ['x', 'y']

# class indicating signal pairs
class_labels = np.array([[['x', 'x'], ['y', 'y']], [['x', 'y'], ['y', 'x']]])


def generate_xor_class(n_data_class, seq_len, cls, s0_ind_mean, s1_ind_mean, noise):
    """
    Generate a class of sequences (two signals are the same for the 0th class and different for the 1st class)
    :param n_data_class: number of sequences in the class
    :param seq_len: length of the sequences
    :param cls: class index (0 or 1)
    :param s0_ind_mean: mean index of the 0th signal
    :param s1_ind_mean: mean index of the 1st signal
    :param noise: noise level (>0)
    :return:
    """
    bk_ind = np.random.randint(0, bk_alphabet_size, size=(n_data_class, seq_len))
    arr = bk_alphabet[bk_ind]
    i_ind = np.arange(n_data_class)
    s0_ind = np.round(s0_ind_mean + np.random.randn(n_data_class) * noise).astype(int)
    s0_ind[s0_ind < 0] = 0
    s1_ind = np.round(s1_ind_mean + np.random.randn(n_data_class) * noise).astype(int)
    s1_ind[s1_ind >= seq_len] = seq_len - 1
    label_ind = np.random.randint(0, 2, n_data_class)

    arr[i_ind, s0_ind] = class_labels[cls][label_ind][:, 0]
    arr[i_ind, s1_ind] = class_labels[cls][label_ind][:, 1]
    if cls == 0:
        classes = np.zeros(n_data_class)
    else:
        classes = np.ones(n_data_class)
    return arr, classes


def generate_xor_data(n_data, seq_len, s0_ind_mean=2, s1_ind_mean=-1, noise=0.0, seed=0):
    """
    Generate the sequences of the XOR distributed classes
    :param n_data: total number of sequences
    :param seq_len: length of the sequences
    :param s0_ind_mean: mean index of the 0th signal
    :param s1_ind_mean: mean index of the 1st signal
    :param noise: noise level (standard deviation from the mean)
    :param seed: random seed
    :return:
    """
    np.random.seed(seed)
    if s1_ind_mean<0:
        s1_ind_mean = seq_len - s0_ind_mean - 1
    n_data_class0 = np.floor(n_data/2).astype(int)
    n_data_class1 = n_data - n_data_class0
    data_class0, classes0 = generate_xor_class(n_data_class0, seq_len, 0, s0_ind_mean, s1_ind_mean, noise)
    data_class1, classes1 = generate_xor_class(n_data_class1, seq_len, 1, s0_ind_mean, s1_ind_mean, noise)
    return np.vstack((data_class0, data_class1)), np.concatenate((classes0, classes1))


def main():
    arr_noiseless, classes_noiseless = generate_xor_data(10, 18)
    print("Sequences with noiseless positions of signals")
    print(arr_noiseless)
    print(classes_noiseless)

    arr_noise, classes_noisy = generate_xor_data(10, 18, noise=0.5)
    print("Sequences with noisy positions of signals")
    print(arr_noise)
    print(classes_noisy)


if __name__ == "__main__":
    main()
