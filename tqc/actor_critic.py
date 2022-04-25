import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
import schnetpack as spk

from tqc import DEVICE


LOG_STD_MIN_MAX = (-20, 2)


class Actor(nn.Module):
    def __init__(self, schnet_args, out_embedding_size, action_scale=0.01):
        super(Actor, self).__init__()
        self.action_scale = action_scale
        self.out_embedding_size = out_embedding_size
        schnet = spk.SchNet(
                        n_interactions=schnet_args["n_interactions"], #3
                        cutoff=schnet_args["cutoff"], #20.0
                        n_gaussians=schnet_args["n_gaussians"] #50
                    )
        output_modules = [ 
                                spk.atomistic.Atomwise(
                                    n_in=schnet.n_atom_basis,
                                    n_out=out_embedding_size * 4,
                                    n_neurons=[out_embedding_size],
                                    contributions='kv'
                                )
                            ]
        self.model = spk.atomistic.model.AtomisticModel(schnet, output_modules)
    
    def forward(self, state_dict):
        kv = self.model(state_dict)['kv']
        k_mu, v_mu, k_sigma, v_sigma = torch.split(kv, self.out_embedding_size, dim=-1)

        # Calculate mean and std of shifts relative to other atoms
        rel_shifts_mean = torch.matmul(k_mu, v_mu.transpose(1, 2))
        rel_shifts_log_std = torch.matmul(k_sigma, v_sigma.transpose(1, 2))

        # Calculate matrix of 1-vectors to other atoms
        P = state_dict['_positions'][:, :, None, :] - state_dict['_positions'][:, None, :, :]
        norm = torch.norm(P, p=2, dim=-1) + 1e-8
        P /= norm[..., None]

        if self.training:
            # Calculate mean and std of actions
            actions_mean = (P * rel_shifts_mean[..., None]).sum(-2)
            actions_log_std = (P * rel_shifts_log_std[..., None]).sum(-2)
            actions_log_std = actions_log_std.clamp(*LOG_STD_MIN_MAX)
            actions_std = torch.exp(actions_log_std)
            # Sample bounded actions and calculate log prob
            tanh_normal = TanhNormal(actions_mean, actions_std)
            actions, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=(1, 2)).unsqueeze(-1)
        else:
            actions = torch.tanh((P * rel_shifts_mean[..., None]).sum(-2))
            log_prob = None

        # Scale actions
        actions *= self.action_scale
        # TODO Check maths !!!
        log_prob =- torch.log([self.action_scale])

        return actions, log_prob

    def select_action(self, state_dict):
        action, _ = self.forward(state_dict)
        return action[0].cpu().detach().numpy()


class Critic(nn.Module):
    def __init__(self, schnet_args, n_nets, n_quantiles, mean=None, stddev=None):
        super(Critic, self).__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        for i in range(self.n_nets):
            schnet = spk.SchNet(
                            n_interactions=schnet_args["n_interactions"], #3
                            cutoff=schnet_args["cutoff"], #20.0
                            n_gaussians=schnet_args["n_gaussians"] #50
                        )
            output_modules = [ 
                                    spk.atomistic.Atomwise(
                                        n_in=schnet.n_atom_basis,
                                        n_out=self.n_quantiles,
                                        property='quantiles',
                                        mean=mean,
                                        stddev=stddev
                                    )
                                ]
            net = spk.atomistic.model.AtomisticModel(schnet, output_modules)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state_dict, actions):
        # To avoid modifying the input
        state_dict_copy = {k:  v.detach().clone() for k, v in state_dict.items()}
        state_dict_copy["_positions"] += actions
        # Schnet changes the state_dict so a deepcopy
        # has to be passed to each net in order to do .backwards()
        quantiles = torch.stack(tuple(
            net({k:  v.detach().clone() for k, v in state_dict_copy.items()})['quantiles'] \
            for net in self.nets
            ), dim=1)
        return quantiles


class TanhNormal(Distribution):
    arg_constraints = {}

    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=DEVICE),
                                      torch.ones_like(self.normal_std, device=DEVICE))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + F.logsigmoid(2 * pre_tanh) + F.logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh
