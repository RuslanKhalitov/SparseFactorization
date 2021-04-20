import torch
from typing import List, Dict
import matplotlib as plt

class Analysis:
    """
    Makes stats for weights, biases, outputs, and gradients of the model
    """
    def __init__(self, active_model, cfg):
        self.active_model = active_model
        self.cfg = cfg
        self.n_g_weights = len(self.cfg['g'])
        self.n_f_weights = len(self.cfg['f'])
        self.n_layers = self.cfg['n_layers'][0]

    def get_params(self):
        """
        Parses model's weights, biases, and the corresponding gradient values
        for both g and fs.
        :return: weights and biases
        """
        # g weights and gradients
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
            'g_weights': g_weights,
            'g_biases': g_biases,
            'g_weights_grads': g_weights_grads,
            'g_biases_grads': g_biases_grads,
            'fs_weights': fs_weights,
            'fs_biases': fs_biases,
            'fs_weights_grads': fs_weights_grads,
            'fs_biases_grads': fs_biases_grads
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
        stats_container = dict.fromkeys(data_dict.keys(), [])

        for var_name, values in data_dict.items():
            if 'grad' in var_name:
                stats_container[var_name] = values.norm().item()
            else:
                stats_container[var_name] = self.stats_calculation(values)

        return stats_container


class PlotGraphs:
    """
    Plots graphs after training
    """
    def __init__(self, train_stats, test_stats, param_stats):
        self.train_stats = train_stats
        self.test_stats = test_stats
        self.param_stats = param_stats

    def plot(self):
        fig, axs = plt.subplots(
            2,
            2,
            subplot_kw=dict(projection="polar"),
            gridspec_kw={'width_ratios': [3, 3, 2, 2, 2, ]},
            sharex='col'
        )

        # Loss
        axs[0, 0].plot(self.train_stats)
        axs[0, 0].set_title('Train')

        axs[0, 1].plot(self.test_stats)
        axs[0, 1].set_title('Test')

        # Gradients
        axs[1, 0].plot(self.param_stats['g_weights_grads'])
        axs[1, 0].set_title('g_weights_grads')

        axs[1, 1].plot(self.param_stats['g_biases_grads'])
        axs[1, 1].set_title('g_biases_grads')

        axs[1, 2].plot(self.param_stats['f_weights_grads'])
        axs[1, 2].set_title('f_weights_grads')

        axs[1, 3].plot(self.param_stats['f_biases_grads'])
        axs[1, 3].set_title('f_biases_grads')

        # Values
        axs[1, 0].plot(self.param_stats['g_weights']['std'])
        axs[1, 0].set_title('g_weights STD')

        axs[1, 1].plot(self.param_stats['g_weights']['mean'])
        axs[1, 1].set_title('g_weights Mean')

        axs[1, 2].plot(self.param_stats['g_weights']['max'])
        axs[1, 2].set_title('g_weights Max')

        # TODO: Finish filling the graph