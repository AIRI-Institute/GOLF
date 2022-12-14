import copy
import numpy as np
import schnetpack as spk
import torch
import torch.nn as nn

from rl.backbones.painn import PaiNN
from rl.actor_critics.tqc import Actor
from utils.utils import ignore_extra_args


backbones = {
    "schnet": ignore_extra_args(spk.SchNet),
    "painn": ignore_extra_args(PaiNN)
}


class Critic(nn.Module):
    def __init__(self, backbone, backbone_args, n_nets, m_nets):
        super(Critic, self).__init__()
        self.m_nets = m_nets
        self.n_nets = n_nets
        self.nets = []

        for i in range(n_nets):
            representation = backbones[backbone](**backbone_args)
            output_modules = [
                spk.atomistic.Atomwise(
                    n_in=representation.n_atom_basis,
                    n_out=1
                )
            ]
            net = spk.atomistic.model.AtomisticModel(representation, output_modules)
            self.add_module(f'qf_{i}', net)
            self.nets.append(net)

    def forward(self, state_dict, actions):
        delta_E_list = []
        for i in range(self.m_nets):
            # Backbone changes the state_dict so a deepcopy
            # has to be passed to each net in order to do .backwards()
            state = {k: v.detach().clone() for k, v in state_dict.items()}
            next_state = {k: v.detach().clone() for k, v in state_dict.items()}
            # Change state here to keep the gradients flowing
            next_state["_positions"] += actions
            delta_E_list.append(self.current_nets[i](state)['y'] - self.current_nets[i](next_state)['y'])

        return torch.stack(delta_E_list, dim=1)
    
    def select_critics(self):
        indices = np.random.choice(self.n_nets, self.m_nets, replace=False)
        self.current_nets = [self.nets[ind] for ind in indices]


class OneStepREDQPolicy(nn.Module):
    def __init__(self, backbone, backbone_args, generate_action_type, out_embedding_size, action_scale_scheduler, 
                 cutoff_type, use_activation, n_nets, m_nets, limit_actions, summation_order):
        super().__init__()
        self.actor = Actor(backbone, backbone_args, generate_action_type, out_embedding_size,
                           action_scale_scheduler, limit_actions, summation_order, cutoff_type, use_activation)
        self.critic = Critic(backbone, backbone_args, n_nets, m_nets)
        self.critic_target = copy.deepcopy(self.critic)

    def act(self, state_dict):
        action, log_prob = self.actor(state_dict)
        self.critic.select_critics()
        Q = self.critic(state_dict, action)
        return Q, action, log_prob
        
    def select_action(self, state_dict):
        return self.actor.select_action(state_dict)
