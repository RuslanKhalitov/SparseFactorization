import os
import torch
from typing import List, Dict
import matplotlib.pyplot as plt


class Analysis:
    """
    Makes stats for weights, biases, outputs, and gradients of the model
    """
    def __init__(self, active_model, cfg, final_dict):
        self.active_model = active_model
        self.cfg = cfg
        self.n_g_weights = len(self.cfg['g'])
        self.n_f_weights = len(self.cfg['f'])
        self.n_layers = self.cfg['n_layers'][0]
        self.final_dict = final_dict

    def get_params(self):
        """
        Parses model's weights, biases, and the corresponding gradient values
        for both g and fs.
        :return: weights and biases
        """
        # g weights and gradients
        with torch.no_grad():
            g_params = self.active_model.g.named_parameters()
        g_weights = []
        g_biases = []
        g_weights_grads = []
        g_biases_grads = []
        for name, param in g_params:
            if 'weight' in name:
                g_weights.append(param.data.flatten())
                g_weights_grads.append(param.grad.flatten())
            else:
                g_biases.append(param.data.flatten())
                g_biases_grads.append(param.grad.flatten())

        g_weights = torch.cat(g_weights)
        g_biases = torch.cat(g_biases)
        g_weights_grads = torch.cat(g_weights_grads)
        g_biases_grads = torch.cat(g_biases_grads)

        # fs weights and gradients
        fs_params = self.active_model.fs.named_parameters()
        fs_weights = []
        fs_biases = []
        fs_weights_grads = []
        fs_biases_grads = []
        for name, param in fs_params:
            if 'weight' in name:
                fs_weights.append(param.data.flatten())
                fs_weights_grads.append(param.grad.flatten())
            else:
                fs_biases.append(param.data.flatten())
                fs_biases_grads.append(param.grad.flatten())

        fs_weights = torch.cat(fs_weights)
        fs_biases = torch.cat(fs_biases)
        fs_weights_grads = torch.cat(fs_weights_grads)
        fs_biases_grads = torch.cat(fs_biases_grads)
        data_dict = {
            'g_weight': g_weights,
            'g_bias': g_biases,
            'grad_g_value': g_weights_grads,
            'grad_g_bias': g_biases_grads,
            'fs_weight': fs_weights,
            'fs_bias': fs_biases,
            'grad_f_value': fs_weights_grads,
            'grad_f_bias': fs_biases_grads
        }
        return data_dict

    @staticmethod
    def stats_calculation(stats: torch.Tensor) -> List[float]:
        return [
            torch.std(stats).item(),
            torch.mean(stats).item(),
            torch.max(stats).item(),
        ]

    def stats_on_params(self):
        data_dict = self.get_params()

        for var_name, values in data_dict.items():
            if 'grad' in var_name:
                self.final_dict[var_name].append(values.norm().item())
            else:
                std_, mean_, max_ = self.stats_calculation(values)
                self.final_dict[var_name+'_std'].append(std_)
                self.final_dict[var_name + '_mean'].append(mean_)
                self.final_dict[var_name + '_max'].append(max_)
        # print(self.final_dict)
        return self.final_dict


class PlotGraphs:
    """
    Plots graphs after training
    """
    def __init__(self, final_dict, cfg):
        self.cfg = cfg
        self.train_stats = final_dict["train_loss"]
        self.test_stats = final_dict["test_loss"]
        self.g_weights_grad = final_dict["grad_g_value"]
        self.g_bias_grad = final_dict["grad_g_bias"]
        self.f_weights_grad = final_dict["grad_f_value"]
        self.f_bias_grad = final_dict["grad_f_bias"]

        self.g_weights_std = final_dict["g_weight_std"]
        self.g_weights_mean = final_dict["g_weight_mean"]
        self.g_weights_max = final_dict["g_weight_max"]
        self.g_bias_std = final_dict["g_bias_std"]
        self.g_bias_mean = final_dict["g_bias_mean"]
        self.g_bias_max = final_dict["g_bias_max"]
        self.fs_weights_std = final_dict["fs_weight_std"]
        self.fs_weights_mean = final_dict["fs_weight_mean"]
        self.fs_weights_max = final_dict["fs_weight_max"]
        self.fs_bias_std = final_dict["fs_bias_std"]
        self.fs_bias_mean = final_dict["fs_bias_mean"]
        self.fs_bias_max = final_dict["fs_bias_max"]

    def plot(self):
        fig, axs = plt.subplots(
            3,
            2,
            sharex=True,
            sharey=False,
            figsize=(10, 10)
        )

        # Loss
        axs[0, 0].plot(self.train_stats)
        axs[0, 0].set_title('Train')
        axs[0, 0].ticklabel_format(useOffset=False)

        axs[0, 1].plot(self.test_stats)
        axs[0, 1].set_title('Test')
        axs[0, 1].ticklabel_format(useOffset=False)

        # Gradients
        axs[1, 0].plot(self.g_weights_grad)
        axs[1, 0].set_title('g_weights_grads')
        axs[1, 0].ticklabel_format(useOffset=False)

        axs[1, 1].plot(self.g_bias_grad)
        axs[1, 1].set_title('g_biases_grads')
        axs[1, 1].ticklabel_format(useOffset=False)

        axs[2, 0].plot(self.f_weights_grad)
        axs[2, 0].set_title('fs_weights_grads')
        axs[2, 0].ticklabel_format(useOffset=False)

        axs[2, 1].plot(self.f_bias_grad)
        axs[2, 1].set_title('fs_biases_grads')
        axs[2, 1].ticklabel_format(useOffset=False)

        # Values
        #axs[2, 0].plot(self.g_weights_std)
        #axs[2, 0].set_title('g_weights STD')

        #axs[2, 1].plot(self.g_weights_mean)
        #axs[2, 1].set_title('g_weights Mean')

        #axs[2, 2].plot(self.g_weights_max)
        #axs[2, 2].set_title('g_weights Max')

        #axs[3, 0].plot(self.g_bias_std)
        #axs[3, 0].set_title('g_bias STD')

        #axs[3, 1].plot(self.g_bias_mean)
        #axs[3, 1].set_title('g_weights Mean')

        #axs[3, 2].plot(self.g_bias_max)
        #axs[3, 2].set_title('g_bias Max')

        #axs[4, 0].plot(self.fs_weights_std)
        #axs[4, 0].set_title('fs_weights STD')

        #axs[4, 1].plot(self.fs_weights_mean)
        #axs[4, 1].set_title('fs_weights Mean')

        #axs[4, 2].plot(self.fs_weights_max)
        #axs[4, 2].set_title('fs_weights Max')

        #axs[5, 0].plot(self.fs_bias_std)
        #axs[5, 0].set_title('fs_bias STD')

        #axs[5, 1].plot(self.fs_bias_mean)
        #axs[5, 1].set_title('fs_weights Mean')

        #axs[5, 2].plot(self.fs_bias_max)
        #axs[5, 2].set_title('fs_bias Max')

        try:
            os.makedirs("SparseFactorization/result_plots")
        except FileExistsError:
            pass
        n_layers = self.cfg['n_layers'][0]
        N = self.cfg['N'][0]
        d = self.cfg['d'][0]
        masking = ~self.cfg['disable_masking'][0]

        plt.savefig(f"SparseFactorization/result_plots/{n_layers}fs_{N}N_{d}d_m{masking}.png")
