from functools import partial

import numpy as np
import schnetpack as spk
import torch
import torch.nn as nn
from schnetpack import properties
from torch.linalg import vector_norm
from torch.optim import LBFGS

from AL import DEVICE
from AL.utils import get_action_scale_scheduler, get_atoms_indices_range, unpad_state
from utils.utils import ignore_extra_args

EPS = 1e-8

backbones = {
    "schnet": ignore_extra_args(spk.representation.SchNet),
    "painn": ignore_extra_args(spk.representation.PaiNN),
}


class Actor(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_args,
        action_scale,
        action_scale_scheduler="Constant",
        action_norm_limit=None,
    ):
        super(Actor, self).__init__()
        self.action_norm_limit = action_norm_limit
        self.action_scale = get_action_scale_scheduler(
            action_scale_scheduler, action_scale
        )

        representation = backbones[backbone](**backbone_args)
        output_modules = [
            spk.atomistic.Atomwise(
                n_in=representation.n_atom_basis,
                n_out=1,
                output_key="energy",
            ),
            spk.atomistic.Forces(energy_key="energy", force_key="anti_gradient"),
        ]

        self.model = spk.model.NeuralNetworkPotential(
            representation=representation,
            input_modules=[spk.atomistic.PairwiseDistances()],
            output_modules=output_modules,
        )

    def _limit_action_norm(self, actions, n_atoms):
        if self.action_norm_limit is None:
            return actions

        coefficient = torch.ones(
            size=(actions.size(0), 1), dtype=torch.float32, device=actions.device
        )
        for molecule_id in range(n_atoms.size(0) - 1):
            max_norm = (
                vector_norm(
                    actions[n_atoms[molecule_id] : n_atoms[molecule_id + 1]],
                    dim=-1,
                    keepdims=True,
                )
                .max(dim=1, keepdims=True)
                .values
            )
            max_norm = torch.maximum(
                max_norm, torch.full_like(max_norm, fill_value=EPS, dtype=torch.float32)
            )
            coefficient[
                n_atoms[molecule_id] : n_atoms[molecule_id + 1]
            ] = torch.minimum(
                self.action_norm_limit / max_norm,
                torch.ones_like(max_norm, dtype=torch.float32),
            )

        return actions * coefficient

    def forward(self, state_dict, t=None, train=False):
        output = self.model(state_dict)
        if train:
            return output
        action_scale = self.action_scale.get(t)
        action = output["anti_gradient"].detach()
        action *= action_scale
        action = self._limit_action_norm(action, get_atoms_indices_range(state_dict))

        return {"action": action, "energy": output["energy"]}


class ALPolicy(nn.Module):
    def __init__(
        self,
        n_parallel,
        backbone,
        backbone_args,
        action_scale,
        action_scale_scheduler,
        max_iter,
        action_norm_limit=None,
        grad_threshold=1e-5,
    ):
        super().__init__()
        self.n_parallel = n_parallel
        self.action_scale = action_scale
        self.max_iter = max_iter
        self.grad_threshold = grad_threshold
        self.optimizer_list = [None] * n_parallel
        self.states = [None] * n_parallel
        self.actor = Actor(
            backbone,
            backbone_args,
            action_scale,
            action_scale_scheduler,
            action_norm_limit,
        )

    def reset(self, initial_states, indices=None):
        if indices is None:
            indices = torch.arange(self.n_parallel)
        unpad_initial_states = unpad_state(initial_states)
        for i, idx in enumerate(indices):
            self.states[idx] = {
                k: v.detach().clone().to(DEVICE)
                for k, v in unpad_initial_states[i].items()
            }
            self.states[idx][properties.R].requires_grad_(True)
            self.optimizer_list[idx] = LBFGS(
                [self.states[idx][properties.R]],
                lr=self.action_scale,
                max_iter=self.max_iter,
            )

    def act(self, t):
        # Save current positions
        prev_positions = [
            self.states[idx][properties.R].detach().clone()
            for idx in range(self.n_parallel)
        ]
        energy = torch.zeros(self.n_parallel, device=DEVICE)

        # Define reevaluation function
        def closure(idx):
            self.optimizer_list[idx].zero_grad()
            # train=True to get correct gradients
            output = self.actor(self.states[idx], t, train=True)
            self.states[idx][properties.R].grad = -output["anti_gradient"].detach()
            energy[idx] = output["energy"]
            return output["energy"]

        # Update all molecules' geometry
        for i, optim in enumerate(self.optimizer_list):
            optim.step(partial(closure, idx=i))

        # Check if optimizers have reached optimal states (for evaluation only)
        done = [
            torch.tensor(
                optim._gather_flat_grad().abs().max() <= self.grad_threshold
            ).unsqueeze(0)
            for optim in self.optimizer_list
        ]

        # Calculate action based on saved positions and resulting geometries
        actions = [
            self.states[idx][properties.R].detach().clone() - prev_positions[idx]
            for idx in range(self.n_parallel)
        ]
        return {
            "action": torch.cat(actions, dim=0),
            "energy": energy,
            "done": torch.cat(done, dim=0),
        }

    def select_action(self, t):
        output = self.act(t)
        action = output["action"].cpu().numpy()
        energy = output["energy"].detach().cpu().numpy()
        done = output["done"].cpu().numpy()
        return action, energy, done
