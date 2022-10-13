import copy
import torch
import torch.nn as nn

import schnetpack as spk
from torch.distributions import Normal
from schnetpack.nn.blocks import MLP

from rl import DEVICE
from rl.backbones.painn import PaiNN
from utils.utils import ignore_extra_args


LOG_STD_MIN_MAX = (-20, 2)


backbones = {
    "schnet": ignore_extra_args(spk.SchNet),
    "painn": ignore_extra_args(PaiNN)
}


class GenerateActionsBlock(nn.Module):
    def __init__(self, out_embedding_size, tanh):
        super().__init__()
        self.out_embedding_size = out_embedding_size
        assert tanh in ["before_projection", "after_projection"],\
            "Variable tanh must take one of two values: {}, {}".format("before_projection", "after_projection")
        self.tanh = tanh

    def forward(self, kv, positions, atoms_mask, action_scale):
        # Mask kv
        kv *= atoms_mask[..., None]
        k_mu, v_mu, actions_log_std = torch.split(kv, [self.out_embedding_size, self.out_embedding_size, 1], dim=-1)
        
        # Calculate mean and std of shifts relative to other atoms
        # Divide by \sqrt(emb_size) to bring initial action means closer to 0
        rel_shifts_mean = torch.matmul(k_mu, v_mu.transpose(1, 2)) / torch.sqrt(torch.FloatTensor([k_mu.size(-1)])).to(DEVICE)
        
        # Bound relative_shifts with tanh if self.tanh == "before_projection"
        if  self.tanh == "before_projection":
            rel_shifts_mean = torch.tanh(rel_shifts_mean)
        
        # Calculate matrix of 1-vectors to other atoms
        P = positions[:, :, None, :] - positions[:, None, :, :]
        norm = torch.norm(P, p=2, dim=-1) + 1e-8
        P /= norm[..., None]
        
        # Project actions
        actions_mean = (P * rel_shifts_mean[..., None]).sum(-2)
        
        # Make actions norm independent of the number of atoms
        actions_mean /= atoms_mask.sum(-1)[:, None, None]
        
        # Bound means with tanh if self.tanh == "after_projection"
        if self.tanh == "after_projection":
            actions_mean = torch.tanh(actions_mean)

        if self.training:
            # Clamp and exp log_std
            actions_log_std = actions_log_std.clamp(*LOG_STD_MIN_MAX)
            actions_std = torch.exp(actions_log_std)
            # Sample actions and calculate log prob
            self.scaled_normal = Normal(actions_mean * action_scale, actions_std * action_scale)
            actions = self.scaled_normal.rsample()
            log_prob = self.scaled_normal.log_prob(actions)
            log_prob *= atoms_mask[..., None]
            log_prob = log_prob.sum(dim=(1, 2)).unsqueeze(-1)
        else:
            actions = action_scale * actions_mean
            log_prob = None
        actions *= atoms_mask[..., None]

        return actions, log_prob

class Actor(nn.Module):
    def __init__(self, backbone, backbone_args, out_embedding_size, action_scale_scheduler, tanh="after_projection"):
        super(Actor, self).__init__()
        self.action_scale_scheduler = action_scale_scheduler

        representation = backbones[backbone](activation='softplus', **backbone_args)
        output_modules = [
            spk.atomistic.Atomwise(
                n_in=representation.n_atom_basis,
                n_out=out_embedding_size * 2 + 1,
                n_neurons=[out_embedding_size],
                contributions='kv'
            )
        ]
        self.model = spk.atomistic.model.AtomisticModel(representation, output_modules)
        self.generate_actions_block = GenerateActionsBlock(out_embedding_size, tanh)
    
    def forward(self, state_dict):
        action_scale = self.action_scale_scheduler.get_action_scale()
        atoms_mask = state_dict['_atom_mask']
        kv = self.model(state_dict)['kv']
        
        actions, log_prob = self.generate_actions_block(kv, state_dict['_positions'], atoms_mask, action_scale)
        return actions, log_prob

    def select_action(self, state_dict):
        action, _ = self.forward(state_dict)
        return action[0].cpu().detach().numpy()


class Critic(nn.Module):
    def __init__(self, backbone, backbone_args, n_nets, out_embedding_size, n_quantiles):
        super(Critic, self).__init__()
        self.nets = []
        self.mlps = []
        self.n_nets = n_nets
        self.n_quantiles = n_quantiles

        for i in range(self.n_nets):
            representation = backbones[backbone](**backbone_args)
            output_modules = [
                spk.atomistic.Atomwise(
                    n_in=representation.n_atom_basis,
                    n_out=out_embedding_size,
                    property='embedding'
                )
            ]
            net = spk.atomistic.model.AtomisticModel(representation, output_modules)
            mlp = MLP(2 * out_embedding_size, n_quantiles)
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

class TQCPolicy(nn.Module):
    def __init__(self, backbone, backbone_args, out_embedding_size, action_scale_scheduler, n_nets, n_quantiles, tanh="after_projection"):
        super().__init__()
        self.actor = Actor(backbone, backbone_args, out_embedding_size, action_scale_scheduler, tanh)
        self.critic = Critic(backbone, backbone_args, n_nets, out_embedding_size, n_quantiles)
        self.critic_target = copy.deepcopy(self.critic)

    def act(self, state_dict):
        action, log_prob = self.actor(state_dict)
        Q = self.critic(state_dict, action)
        return Q, action, log_prob
        
    def select_action(self, state_dict):
        return self.actor.select_action(state_dict)
