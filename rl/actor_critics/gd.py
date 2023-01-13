import torch
import torch.nn as nn

import schnetpack as spk
from torch.linalg import vector_norm

from utils.utils import ignore_extra_args

EPS = 1e-8


class Actor(nn.Module):
    def __init__(self, backbone_args, action_scale, action_norm_limit=None):
        super(Actor, self).__init__()
        self.action_norm_limit = action_norm_limit
        self.action_scale = action_scale

        representation = ignore_extra_args(spk.SchNet)(**backbone_args)
        output_modules = [
            spk.atomistic.Atomwise(
                n_in=representation.n_atom_basis,
                n_out=1,
                property='energy',
                negative_dr=True,
                derivative='anti_gradient'
            )
        ]
        self.model = spk.atomistic.model.AtomisticModel(representation, output_modules)

    def _limit_action_norm(self, actions):
        if self.action_norm_limit is None:
            return actions

        max_norm = vector_norm(actions, dim=-1, keepdims=True).max(dim=1, keepdims=True).values
        max_norm = torch.maximum(max_norm, torch.full_like(max_norm, fill_value=EPS, dtype=torch.float32))
        coefficient = torch.minimum(self.action_norm_limit / max_norm, torch.ones_like(max_norm, dtype=torch.float32))
        actions *= coefficient

        return actions

    def forward(self, state_dict):
        atoms_mask = state_dict['_atom_mask']
        output = self.model(state_dict)
        action = output['anti_gradient'].detach() * atoms_mask.unsqueeze(-1)
        action *= self.action_scale
        action = self._limit_action_norm(action)

        return {'action': action, 'energy': output['energy']}

    def select_action(self, state_dict):
        action = self.forward(state_dict)['action']
        return action.cpu().numpy()


class GDPolicy(nn.Module):
    def __init__(self, backbone_args, action_scale, action_norm_limit=None):
        super().__init__()
        self.actor = Actor(backbone_args, action_scale, action_norm_limit)

    def act(self, state_dict):
        output = self.actor(state_dict)
        return output

    def select_action(self, state_dict):
        return self.actor.select_action(state_dict)
