import copy
import torch
import torch.nn as nn

import schnetpack as spk
import schnetpack.nn as snn
from schnetpack.nn.blocks import MLP

from rl.backbones.painn import PaiNN
from rl.generate_action_block import GenerateActionsBlock
from utils.utils import ignore_extra_args


backbones = {
    "schnet": ignore_extra_args(spk.SchNet),
    "painn": ignore_extra_args(PaiNN)
}


class Actor(nn.Module):
    def __init__(self, backbone, backbone_args, out_embedding_size, action_scale_scheduler,
                 limit_actions, summation_order):
        super(Actor, self).__init__()
        self.action_scale_scheduler = action_scale_scheduler

        self.cutoff_network = snn.get_cutoff_by_string('hard')(backbone_args['cutoff'])
        representation = backbones[backbone](**backbone_args)
        output_modules = [
            spk.atomistic.Atomwise(
                n_in=representation.n_atom_basis,
                n_out=out_embedding_size * 2 + 1,
                n_neurons=[out_embedding_size],
                contributions='kv'
            )
        ]
        self.model = spk.atomistic.model.AtomisticModel(representation, output_modules)
        self.generate_actions_block = GenerateActionsBlock(out_embedding_size, limit_actions,
                                                           self.cutoff_network, summation_order)
    
    def forward(self, state_dict):
        action_scale = self.action_scale_scheduler.get_action_scale()
        atoms_mask = state_dict['_atom_mask']
        kv = self.model(state_dict)['kv']
        
        actions, log_prob = self.generate_actions_block(kv, state_dict['_positions'], atoms_mask, action_scale)
        return actions, log_prob

    def select_action(self, state_dict):
        action, _ = self.forward(state_dict)
        return action.cpu().detach().numpy()


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
    def __init__(self, backbone, backbone_args, out_embedding_size, action_scale_scheduler, 
                 n_nets, n_quantiles, limit_actions, summation_order):
        super().__init__()
        self.actor = Actor(backbone, backbone_args, out_embedding_size, action_scale_scheduler,
                           limit_actions, summation_order)
        self.critic = Critic(backbone, backbone_args, n_nets, out_embedding_size, n_quantiles)
        self.critic_target = copy.deepcopy(self.critic)

    def act(self, state_dict):
        action, log_prob = self.actor(state_dict)
        Q = self.critic(state_dict, action)
        return Q, action, log_prob
        
    def select_action(self, state_dict):
        return self.actor.select_action(state_dict)
