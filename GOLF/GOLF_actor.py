import asyncio
import collections

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from GOLF import DEVICE
from GOLF.models import DimeNetPlusPlusPotential
from GOLF.optim import lbfgs
from GOLF.utils import (
    get_atoms_indices_range,
    get_conformation_lr_scheduler,
    get_n_atoms,
)
from utils.utils import ignore_extra_args

KCALMOL_2_HARTREE = 627.5

EPS = 1e-8

NeuralNetworkPotentials = {
    "DimenetPlusPlus": ignore_extra_args(DimeNetPlusPlusPotential),
}


class Actor(nn.Module):
    def __init__(
        self,
        nnp_type,
        nnp_args,
        forces_norm_limit=None,
    ):
        super(Actor, self).__init__()
        self.forces_norm_limit = forces_norm_limit

        self.nnp = NeuralNetworkPotentials[nnp_type](**nnp_args)

        self.last_energy = None
        self.last_forces = None

    def _limit_forces_norm(self, forces, n_atoms):
        if self.forces_norm_limit is None:
            return forces

        coefficient = torch.ones(
            size=(forces.size(0), 1), dtype=torch.float32, device=forces.device
        )
        for molecule_id in range(n_atoms.size(0) - 1):
            max_norm = (
                torch.linalg.vector_norm(
                    forces[n_atoms[molecule_id] : n_atoms[molecule_id + 1]],
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
                self.forces_norm_limit / max_norm,
                torch.ones_like(max_norm, dtype=torch.float32),
            )

        return forces * coefficient

    def _save_last_output(self, energy, forces):
        self.last_energy = energy.detach().clone()
        self.last_forces = forces.detach().clone()

    def _get_last_output(self):
        if self.last_energy is None or self.last_forces is None:
            raise ValueError("Last output has not been set yet!")
        return self.last_energy, self.last_forces

    def forward(self, batch, active_optimizers_ids=None, train=False):
        _, _, energy, forces = self.nnp(
            pos=batch.pos, atom_z=batch.z, batch_mapping=batch.batch
        )
        self._save_last_output(energy, forces)
        if train:
            return {"forces": forces.detach(), "energy": energy}
        forces = forces.detach()
        forces = self._limit_forces_norm(forces, get_atoms_indices_range(batch))

        return {"forces": forces, "energy": energy}


class RdkitActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def forward(self, batch, active_optimizers_ids=None, train=False):
        if active_optimizers_ids is None:
            opt_ids = list(range(self.env.n_parallel))
        else:
            opt_ids = active_optimizers_ids

        # Update atoms inside env
        current_coordinates = [
            self.env.unwrapped.atoms[idx].get_positions() for idx in opt_ids
        ]
        n_atoms = get_n_atoms(batch)
        new_coordinates = torch.split(
            batch.pos.detach().cpu(),
            n_atoms.tolist(),
        )
        assert len(new_coordinates) == len(opt_ids)
        new_coordinates = [
            np.float64(new_coordinates[i].numpy()) for i in range(len(opt_ids))
        ]
        # Update coordinates inside env
        self.env.rdkit_oracle.update_coordinates(new_coordinates, indices=opt_ids)
        _, energies, forces = self.env.rdkit_oracle.calculate_energies_forces(
            indices=opt_ids
        )

        # Restore original coordinates
        self.env.rdkit_oracle.update_coordinates(current_coordinates, indices=opt_ids)

        # Forces in (kcal/mol)/angstrom. Transform into hartree/angstrom.
        forces = torch.cat(
            [torch.tensor(force / KCALMOL_2_HARTREE) for force in forces]
        )

        return {"forces": forces, "energy": torch.tensor(energies)}


class ConformationOptimizer(nn.Module):
    def __init__(
        self,
        n_parallel,
        actor,
        lr_scheduler,
        t_max,
        optimizer,
        optimizer_kwargs,
    ):
        super().__init__()
        self.n_parallel = n_parallel
        self.lr_scheduler = get_conformation_lr_scheduler(
            lr_scheduler, optimizer_kwargs["lr"], t_max
        )
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_list = [None] * n_parallel
        self.states = [None] * n_parallel
        self.actor = actor

    def reset(self, initial_batch, indices=None):
        if indices is None:
            indices = torch.arange(self.n_parallel)
        unpad_initial_states = initial_batch.to_data_list()
        for i, idx in enumerate(indices):
            self.states[idx] = unpad_initial_states[i].copy()
            self.states[idx].pos.requires_grad_(True)
            self.optimizer_list[idx] = self.optimizer(
                [self.states[idx].pos], **self.optimizer_kwargs
            )

    def act(self, t):
        # Update learning rate
        lrs = self.lr_scheduler.get(t)
        for idx, optim in enumerate(self.optimizer_list):
            for g in optim.param_groups:
                g["lr"] = lrs[idx]

        # Save current positions
        prev_positions = [
            self.states[idx].pos.detach().clone() for idx in range(self.n_parallel)
        ]
        energy = torch.zeros(self.n_parallel)

        for optim in self.optimizer_list:
            optim.zero_grad()

        # Compute forces
        batch = Batch.from_data_list(self.states).to(DEVICE)
        n_atoms = get_n_atoms(batch)
        output = self.actor(batch, train=True)
        energy = output["energy"]
        gradients = torch.split(
            -output["forces"].detach(),
            n_atoms.tolist(),
        )

        # Update all molecules' geometry
        for idx, optim in enumerate(self.optimizer_list):
            self.states[idx].pos.grad = gradients[idx].to(DEVICE)
            optim.step()

        # Done always False
        done = [torch.tensor([False]) for _ in self.optimizer_list]

        # Calculate action based on saved positions and resulting geometries
        actions = [
            self.states[idx].pos.detach().clone() - prev_positions[idx]
            for idx in range(self.n_parallel)
        ]
        is_finite_action = [
            torch.isfinite(action).all().unsqueeze(dim=0) for action in actions
        ]
        return {
            "action": torch.cat(actions, dim=0),
            "energy": energy.detach(),
            "done": torch.cat(done, dim=0),
            "n_iter": torch.ones_like(energy),
            "is_finite_action": torch.cat(is_finite_action),
            "forces": output["forces"].detach(),
        }

    def select_action(self, t):
        output = self.act(t)
        return {key: value.cpu().numpy() for key, value in output.items()}


class AsyncLBFGS:
    def __init__(
        self,
        state: dict,
        policy2optimizer_queue: asyncio.Queue,
        optimizer2policy_queue: asyncio.Queue,
        optimizer_kwargs: dict,
        grad_threshold: float,
    ):
        super().__init__()
        self.state = state
        self.state.pos.requires_grad_(True)
        self.policy2optimizer_queue = policy2optimizer_queue
        self.optimizer2policy_queue = optimizer2policy_queue
        self.optimizer = lbfgs.LBFGS([self.state.pos], **optimizer_kwargs)
        self.energy = None
        self.forces = None
        self.n_iter = None
        self.grad_threshold = grad_threshold

    async def closure(self):
        self.optimizer.zero_grad()
        await self.optimizer2policy_queue.put(self.state)
        forces, energy = await self.policy2optimizer_queue.get()
        self.state.pos.grad = -forces
        # Energy and forces before step
        if self.n_iter == 0:
            self.forces = forces
            self.energy = energy

        self.n_iter += 1
        return energy

    async def step(self):
        self.n_iter = 0
        previous_position = self.state.pos.detach().clone()
        await self.optimizer.step(self.closure)
        await self.optimizer2policy_queue.put(None)
        done = torch.unsqueeze(
            self.optimizer._gather_flat_grad().abs().max() <= self.grad_threshold, dim=0
        )
        action = self.state.pos.detach().clone() - previous_position
        is_finite_action = torch.isfinite(action).all().unsqueeze(dim=0)
        return {
            "action": action,
            "energy": self.energy,
            "done": done,
            "is_finite_action": is_finite_action,
            "n_iter": torch.tensor([self.n_iter]),
            "forces": self.forces,
        }


class LBFGSConformationOptimizer(nn.Module):
    def __init__(
        self,
        n_parallel,
        actor,
        optimizer_kwargs,
        grad_threshold=1e-5,
        lbfgs_device="cuda",
    ):
        super().__init__()
        self.n_parallel = n_parallel
        self.grad_threshold = grad_threshold
        self.optimizer_kwargs = optimizer_kwargs
        self.actor = actor
        self.lbfgs_device = torch.device(lbfgs_device)
        self.loop = asyncio.new_event_loop()
        self.policy2optimizer_queues = None
        self.optimizer2policy_queues = None
        self.loop.run_until_complete(self.set_queues())
        self.conformation_optimizers = [None] * self.n_parallel

    async def set_queues(self):
        self.policy2optimizer_queues = [
            asyncio.Queue(maxsize=1) for _ in range(self.n_parallel)
        ]
        self.optimizer2policy_queues = [
            asyncio.Queue(maxsize=1) for _ in range(self.n_parallel)
        ]

    def reset(self, initial_states, indices=None):
        if indices is None:
            indices = torch.arange(self.n_parallel)
        unpad_initial_states = initial_states.to_data_list()
        torch.set_grad_enabled(True)
        for i, idx in enumerate(indices):
            state = unpad_initial_states[i].copy().to(self.lbfgs_device)
            self.conformation_optimizers[idx] = AsyncLBFGS(
                state,
                self.policy2optimizer_queues[idx],
                self.optimizer2policy_queues[idx],
                self.optimizer_kwargs,
                self.grad_threshold,
            )

    async def _act_task(self):
        conformation_optimizers_ids = set(range(self.n_parallel))
        while True:
            individual_states = {}
            stopped_optimizers_ids = set()
            for conformation_optimizer_id in conformation_optimizers_ids:
                individual_state = await self.optimizer2policy_queues[
                    conformation_optimizer_id
                ].get()
                if individual_state is None:
                    stopped_optimizers_ids.add(conformation_optimizer_id)
                    continue

                individual_states[conformation_optimizer_id] = individual_state

            conformation_optimizers_ids -= stopped_optimizers_ids
            if len(individual_states) == 0:
                break

            batch = Batch.from_data_list(list(individual_states.values())).to(DEVICE)
            n_atoms = get_n_atoms(batch)
            torch.set_grad_enabled(True)
            output = self.actor(
                state_dict=batch,
                active_optimizers_ids=list(conformation_optimizers_ids),
                train=True,
            )
            forces = torch.split(
                output["forces"].detach().to(self.lbfgs_device),
                n_atoms.tolist(),
            )
            energies = output["energy"].detach().to(self.lbfgs_device).view(-1, 1)
            for i, optimizer_idx in enumerate(individual_states.keys()):
                await self.policy2optimizer_queues[optimizer_idx].put(
                    (forces[i], energies[i])
                )

    async def _act_async(self):
        tasks = [
            conformation_optimizer.step()
            for conformation_optimizer in self.conformation_optimizers
        ]
        tasks.append(self._act_task())
        task_results = await asyncio.gather(*tasks)

        result = collections.defaultdict(list)
        for task in task_results[:-1]:
            for key, value in task.items():
                result[key].append(value)

        for key, value in result.items():
            result[key] = torch.cat(value, dim=0)

        return result

    def act(self, t):
        return self.loop.run_until_complete(self._act_async())

    def select_action(self, t):
        output = self.act(t)

        return {key: value.cpu().numpy() for key, value in output.items()}
