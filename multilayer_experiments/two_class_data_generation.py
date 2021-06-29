"""
Provides generating functions for two-class data
"""
import random
import torch
import matplotlib.pyplot as plt


def _generate_binary_from_gaussian(n_seq, len_seq, mu, sigma):
    """
    Generate binary data according to a non-normalized Gaussian density
    :param n_seq: number of sequences
    :param len_seq: length of sequence
    :param mu: mean of Gaussian
    :param sigma: std of Gaussian
    :return: data tensor of size n_seq x len_seq
    """
    pos = torch.arange(len_seq).repeat(n_seq, 1)
    threshold = 1 - torch.exp(-(pos - mu) ** 2 / (2 * sigma * sigma))
    ran = torch.rand(n_seq, len_seq)
    return (ran > threshold).float()


def _generate_binary_uniform(n_seq, len_seq, thredshold):
    """
    Generate binary data according to the given threshold
    :param n_seq: number of sequences
    :param len_seq: length of sequence
    :param thredshold: yields 1 if rand > threshold, otherwise yields 0
    :return: data tensor of size n_seq x len_seq
    """
    ran = torch.rand(n_seq, len_seq)
    return (ran > thredshold).float()


def _generate_gaussian_curve(n_seq, len_seq, mu, sigma):
    """
    generate non-normalized Gaussian curve data
    :param n_seq: number of sequences
    :param len_seq: length of sequence
    :param mu: mean of Gaussian
    :param sigma: std of Gaussian
    :return: data tensor of size n_seq x len_seq
    """
    return torch.exp(-(torch.arange(len_seq).repeat(n_seq, 1) - mu) ** 2 / (2 * sigma * sigma))


def generate_two_class_data(n_seq, len_seq, binary=True, same_sigma=True, xor=False,
                            threshold=0.99, noise_level=0.1):
    """
    Generate two-class data for classification
    :param n_seq: number of sequences
    :param len_seq: length of sequence
    :param binary: boolean, to generate binary or curve data
    :param same_sigma: boolean, use same sigma or different sigmas
    :param xor: boolean, to generate simple or xor distributed classes
    :param threshold: between 0 and 1, used in binary data generation to threshold continuous data to binary
    :param noise_level: between 0 and 1, used in curve data generation as Gaussian noise std
    :return: (data, class_labels) where \
    data is tensor of size n_seq x len_seq \
    class_labels is tensor of size n_seq
    """
    if binary:
        gen_func = _generate_binary_from_gaussian
    else:
        gen_func = _generate_gaussian_curve

    sigma0 = len_seq / 15
    if same_sigma:
        sigma1 = sigma0
    else:
        sigma1 = len_seq / 45
    mu0 = len_seq / 4
    mu1 = len_seq / 4 * 3


    if binary:
        noise = _generate_binary_uniform(n_seq, len_seq, threshold)
    else:
        noise = torch.randn(n_seq, len_seq) * noise_level

    if xor:
        data0a = gen_func(n_seq // 4, len_seq, mu0, sigma0)
        data0b = gen_func(n_seq // 4, len_seq, mu1, sigma1)
        data1a_p0 = gen_func(n_seq // 4, len_seq, mu0, sigma0)
        data1a_p1 = gen_func(n_seq // 4, len_seq, mu1, sigma1)
        data1b = torch.zeros(n_seq // 4, len_seq)
        if binary:
            data1a = (data1a_p0 + data1a_p1 > 0).float()
            data = (torch.vstack((data0a, data0b, data1a, data1b)) + noise > 0).float()
        else:
            data1a = data1a_p0 + data1a_p1
            data = torch.vstack((data0a, data0b, data1a, data1b)) + noise
    else:
        data0 = gen_func(n_seq // 2, len_seq, mu0, sigma0)
        data1 = gen_func(n_seq // 2, len_seq, mu1, sigma1)
        data = torch.vstack((data0, data1)) + noise

    if binary:
        data[data < 0] = 0
        data[data > 1] = 1

    class_labels_0 = torch.zeros(n_seq // 2)
    class_labels_1 = torch.ones(n_seq // 2)
    class_labels = torch.cat((class_labels_0, class_labels_1))
    return data, class_labels


def generate_two_class_mixed_data(n_seq, len_seq, binary=True, same_sigma=True, xor=False,
                                  threshold=0.99, noise_level=0.1, sparsity=0.4):
    """
    Generate two-class data which are linearly mixed from simple distributions
    :param n_seq: number of sequences
    :param len_seq: length of sequence
    :param binary: boolean, to generate binary or curve data
    :param same_sigma: boolean, use same sigma or different sigmas
    :param xor: boolean, to generate simple or xor distributed classses
    :param threshold: between 0 and 1, used in binary data generation to threshold continuous data to binary
    :param noise_level: between 0 and 1, used in curve data generation as Gaussian noise std
    :param sparsity: between 0 and 1, controls the sparsity of the interaction matrix
    :return: (data, class_labels, data_orig, interaction, mixing) where \
    data is tensor of size n_seq x len_seq \
    class_labels is tensor of size n_seq \
    data_orig is the data tensor before mixing, of size n_seq x len_seq \
    interaction is the ground truth interaction matrix, of size len_seq x len_seq \
    mixing is the mixing matrix, which is inverse of interaction and of size len_seq x len_seq
    """
    data_orig, class_labels = generate_two_class_data(n_seq, len_seq, binary, same_sigma, xor, threshold, noise_level)
    interaction = torch.rand(len_seq, len_seq) - 0.5
    interaction[torch.abs(interaction) < (1 - sparsity) / 2] = 0
    mixing = torch.inverse(interaction)
    data = data_orig @ mixing
    return data, class_labels, data_orig, interaction, mixing


def generate_n_gaussians(n_seq, len_seq, n_gaussians, noise_level=0.1):

    gen_func = _generate_gaussian_curve
    assert n_gaussians > 0, 'Please set n_gaussians as int > 0'
    assert n_seq % n_gaussians == 0, 'Number of sequences should be divisible by n_gaussians'
    labels = []
    data = []
    for i in range(n_seq):
        ngaus = random.randint(0, n_gaussians)
        # Starts with some noise
        sequence = torch.randn(1, len_seq) * noise_level
        for j in range(ngaus):
            sigma = len_seq / 10 * random.random()
            mu = random.random() * len_seq
            sequence += gen_func(1, len_seq, mu, sigma)

        data.append(sequence)
        labels.append(float(ngaus))

    data = torch.vstack(data)
    labels = torch.tensor(labels)

    return data, labels


def main():
    """ simple test """
    data, class_label = generate_two_class_data(4, 100, binary=False, same_sigma=True, xor=False)

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(data[i], '.')
    plt.show()


if __name__ == "__main__":
    main()
