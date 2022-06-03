import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
import schnetpack as spk

from tqc import DEVICE


LOG_STD_MIN_MAX = (-20, 2)


class Actor(nn.Module):
    def __init__(self, schnet_args, out_embedding_size, action_scale=1.0):
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
                                    n_out=out_embedding_size * 2 + 3,
                                    n_neurons=[out_embedding_size],
                                    contributions='kv'
                                )
                            ]
        self.model = spk.atomistic.model.AtomisticModel(schnet, output_modules)
    
    def forward(self, state_dict, return_relative_shifts=False):
        kv = self.model(state_dict)['kv']
        k_mu, v_mu, actions_log_std = torch.split(kv, [self.out_embedding_size, self.out_embedding_size, 3], dim=-1)
        # Calculate mean and std of shifts relative to other atoms
        rel_shifts_mean = torch.matmul(k_mu, v_mu.transpose(1, 2))

        # Calculate matrix of 1-vectors to other atoms
        P = state_dict['_positions'][:, :, None, :] - state_dict['_positions'][:, None, :, :]
        norm = torch.norm(P, p=2, dim=-1) + 1e-8
        P /= norm[..., None]
        # Project actions
        actions_mean = (P * rel_shifts_mean[..., None]).sum(-2)

        if self.training:
            # Clamp and exp log_std
            actions_log_std = actions_log_std.clamp(*LOG_STD_MIN_MAX)
            actions_std = torch.exp(actions_log_std)
            # Sample bounded actions and calculate log prob
            tanh_normal = TanhNormal(actions_mean, actions_std)
            actions, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=(1, 2)).unsqueeze(-1)
        else:
            actions = torch.tanh(actions_mean)
            log_prob = None
        if return_relative_shifts:
            return actions, log_prob, rel_shifts_mean, P
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
        quantiles_list = []
        for net in self.nets:
            # Schnet changes the state_dict so a deepcopy
            # has to be passed to each net in order to do .backwards()
            next_state = {k:  v.detach().clone() for k, v in state_dict.items()}
            # Change state here to keep the gradients flowing
            next_state["_positions"] += actions
            quantiles_list.append(net(next_state)['quantiles'])
        quantiles = torch.stack(quantiles_list, dim=1)
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
