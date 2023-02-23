import torch
import torch.nn as nn

import schnetpack as spk
from torch.linalg import vector_norm

from AL.utils import get_atoms_indices_range, get_action_scale_scheduler
from utils.utils import ignore_extra_args

EPS = 1e-8

backbones = {
    "schnet": ignore_extra_args(spk.representation.SchNet),
    "painn": ignore_extra_args(spk.representation.PaiNN)
}


class Actor(nn.Module):
    def __init__(self, backbone, backbone_args, action_scale,
                 action_scale_sheduler="Constant", action_norm_limit=None):
        super(Actor, self).__init__()
        self.action_norm_limit = action_norm_limit
        self.action_scale = get_action_scale_scheduler(action_scale_sheduler, action_scale)

        representation = backbones[backbone](**backbone_args)
        output_modules = [
            spk.atomistic.Atomwise(
                n_in=representation.n_atom_basis,
                n_out=1,
                output_key='energy',
            ),
            spk.atomistic.Forces(energy_key='energy', force_key='anti_gradient'),
        ]

        self.model = spk.model.NeuralNetworkPotential(
            representation=representation,
            input_modules=[spk.atomistic.PairwiseDistances()],
            output_modules=output_modules
        )

    def _limit_action_norm(self, actions, n_atoms):
        if self.action_norm_limit is None:
            return actions

        coefficient = torch.ones(size=(actions.size(0), 1), dtype=torch.float32, device=actions.device)
        for molecule_id in range(n_atoms.size(0) - 1):
            max_norm = vector_norm(actions[n_atoms[molecule_id]:n_atoms[molecule_id + 1]], dim=-1, keepdims=True)\
                .max(dim=1, keepdims=True).values
            max_norm = torch.maximum(max_norm, torch.full_like(max_norm, fill_value=EPS, dtype=torch.float32))
            coefficient[n_atoms[molecule_id]:n_atoms[molecule_id + 1]] = \
                torch.minimum(self.action_norm_limit / max_norm, torch.ones_like(max_norm, dtype=torch.float32))

        return actions * coefficient

    def forward(self, state_dict, t=None, train=False):
        output = self.model(state_dict)
        if train:
            return output

        action_scale = self.action_scale.get(t)
        action = output['anti_gradient'].detach()
        action *= action_scale
        action = self._limit_action_norm(action, get_atoms_indices_range(state_dict))

        return {'action': action, 'energy': output['energy']}

    def select_action(self, state_dict, t):
        output = self.forward(state_dict, t)
        action = output['action'].cpu().numpy()
        energy = output['energy'].detach().cpu().numpy()
        return action, energy


class ALPolicy(nn.Module):
    def __init__(self, backbone, backbone_args, action_scale,
                 action_scale_sheduler, action_norm_limit=None):
        super().__init__()
        self.actor = Actor(backbone, backbone_args, action_scale,
                           action_scale_sheduler, action_norm_limit)

    def act(self, state_dict, t):
        return self.actor(state_dict, t)

    def select_action(self, state_dict, t):
        return self.actor.select_action(state_dict, t)
