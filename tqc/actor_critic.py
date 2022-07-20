import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack as spk

from math import floor
from torch.distributions import Distribution, Normal
from schnetpack.nn.blocks import MLP

from tqc import DEVICE
#from tqc.schnet import SchNet, AtomisticModel


LOG_STD_MIN_MAX = (-20, 2)


class Actor(nn.Module):
    def __init__(self, schnet_args, out_embedding_size, action_scale_scheduler):
        super(Actor, self).__init__()
        self.action_scale_scheduler = action_scale_scheduler
        self.out_embedding_size = out_embedding_size
        # SchNet backbone is shared between actor and all critics
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
        action_scale = self.action_scale_scheduler.get_action_scale()
        kv = self.model(state_dict)['kv']
        k_mu, v_mu, actions_log_std = torch.split(kv, [self.out_embedding_size, self.out_embedding_size, 3], dim=-1)
        # Calculate mean and std of shifts relative to other atoms
        rel_shifts_mean = torch.matmul(k_mu, v_mu.transpose(1, 2)) / torch.sqrt(torch.FloatTensor([k_mu.size(-1)]))

        # Calculate matrix of 1-vectors to other atoms
        P = state_dict['_positions'][:, :, None, :] - state_dict['_positions'][:, None, :, :]
        norm = torch.norm(P, p=2, dim=-1) + 1e-8
        P /= norm[..., None]
        # Project actions
        actions_mean = (P * rel_shifts_mean[..., None]).sum(-2)
        # Bound means with tanh
        actions_mean = torch.tanh(actions_mean)

        if self.training:
            # Clamp and exp log_std
            actions_log_std = actions_log_std.clamp(*LOG_STD_MIN_MAX)
            actions_std = torch.exp(actions_log_std)
            self.actions_std = actions_std
            # Sample actions and calculate log prob
            self.scaled_normal = Normal(actions_mean * action_scale, actions_std * action_scale)
            actions = self.scaled_normal.rsample()
            log_prob = self.scaled_normal.log_prob(actions)
            log_prob = log_prob.sum(dim=(1, 2)).unsqueeze(-1)
        else:
            actions = action_scale * actions_mean
            log_prob = None
        
        if return_relative_shifts:
            return actions, log_prob, rel_shifts_mean, P
        return actions, log_prob

    def select_action(self, state_dict):
        action, _ = self.forward(state_dict)
        return action[0].cpu().detach().numpy()


class Critic(nn.Module):
    def __init__(self, schnet_args, n_nets, schnet_out_embedding_size, n_quantiles, mean=None, stddev=None):
        super(Critic, self).__init__()
        self.nets = []
        self.mlps = []
        self.n_nets = n_nets
        self.schnet_out_embedding_size = schnet_out_embedding_size
        self.n_quantiles = n_quantiles
        for i in range(self.n_nets):
            schnet = spk.SchNet(
                            n_interactions=schnet_args["n_interactions"], #3
                            cutoff=schnet_args["cutoff"], #20.0
                            n_gaussians=schnet_args["n_gaussians"] #50
                        )
            output_modules = [ 
                                    spk.atomistic.Atomwise(
                                        n_in=schnet.n_atom_basis,
                                        n_out=self.schnet_out_embedding_size,
                                        property='embedding',
                                        mean=mean,
                                        stddev=stddev
                                    )
                                ]
            net = spk.atomistic.model.AtomisticModel(schnet, output_modules)
            mlp = MLP(2 * self.schnet_out_embedding_size, self.n_quantiles)
            self.add_module(f'qf_{i}', net)
            self.add_module(f'mlp_{i}', mlp)
            self.nets.append(net)
            self.mlps.append(mlp)

    def forward(self, state_dict, actions):
        quantiles_list = []
        for i in range(self.n_nets):
            # Schnet changes the state_dict so a deepcopy
            # has to be passed to each net in order to do .backwards()
            state = {k: v.detach().clone() for k, v in state_dict.items()}
            next_state = {k: v.detach().clone() for k, v in state_dict.items()}
            # Change state here to keep the gradients flowing
            next_state["_positions"] += actions
            state_emb = self.nets[i](state)['embedding']
            next_state_emb = self.nets[i](next_state)['embedding']
            quantiles_list.append(self.mlps[i](torch.cat((state_emb, next_state_emb), dim=-1)))
        quantiles = torch.stack(quantiles_list, dim=1)
        return quantiles


class ScaledNormal(Distribution):
    arg_constraints = {}

    def __init__(self, normal_mean, normal_std, scale=1.0):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.scale = torch.FloatTensor([scale]).to(DEVICE)
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=DEVICE),
                                      torch.ones_like(self.normal_std, device=DEVICE))
        self.normal = Normal(normal_mean, normal_std)
    
    def log_prob(self, value):
        log_det = torch.log(self.scale)
        return self.normal.log_prob(value) - log_det

    def rsample(self):
        value = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return self.scale * value, value
