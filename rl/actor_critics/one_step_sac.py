import schnetpack as spk
import torch.nn as nn

from rl.backbones.painn import PaiNN
from rl.actor_critics.tqc import Actor
from utils.utils import ignore_extra_args


backbones = {
    "schnet": ignore_extra_args(spk.SchNet),
    "painn": ignore_extra_args(PaiNN)
}


class Critic(nn.Module):
    def __init__(self, backbone, backbone_args):
        super(Critic, self).__init__()
        representation = backbones[backbone](**backbone_args)
        output_modules = [
            spk.atomistic.Atomwise(
                n_in=representation.n_atom_basis,
                n_out=1
            )
        ]
        self.net = spk.atomistic.model.AtomisticModel(representation, output_modules)

    def forward(self, state_dict, actions):
        # Backbone changes the state_dict so a deepcopy
        # has to be passed to each net in order to do .backwards()
        state = {k: v.detach().clone() for k, v in state_dict.items()}
        next_state = {k: v.detach().clone() for k, v in state_dict.items()}

        # Change state here to keep the gradients flowing
        next_state["_positions"] += actions
        E_state = self.net(state)['y']
        E_next_state = self.net(next_state)['y']

        return E_state - E_next_state

class OneStepSACPolicy(nn.Module):
    def __init__(self, backbone, backbone_args, generate_action_type, out_embedding_size, action_scale_scheduler, 
                 cutoff_type, use_activation, limit_actions, summation_order):
        super().__init__()
        self.actor = Actor(backbone, backbone_args, generate_action_type, out_embedding_size,
                           action_scale_scheduler, limit_actions, summation_order, cutoff_type, use_activation)
        self.critic = Critic(backbone, backbone_args)

    def act(self, state_dict):
        action, log_prob = self.actor(state_dict)
        Q = self.critic(state_dict, action)
        return Q, action, log_prob
        
    def select_action(self, state_dict):
        return self.actor.select_action(state_dict)
