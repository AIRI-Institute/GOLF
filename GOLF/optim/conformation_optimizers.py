import asyncio
import collections

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from GOLF import DEVICE
from GOLF.optim import LBFGS
from GOLF.utils import get_n_atoms


class ConformationOptimizer(nn.Module):
    def __init__(
        self,
        n_parallel,
        actor,
        optimizer,
        optimizer_kwargs,
    ):
        super().__init__()
        self.n_parallel = n_parallel
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
            self.states[idx] = unpad_initial_states[i].clone()
            self.states[idx].pos.requires_grad_(True)
            self.optimizer_list[idx] = self.optimizer(
                [self.states[idx].pos], **self.optimizer_kwargs
            )

    def act(self):
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
        self.optimizer = LBFGS([self.state.pos], **optimizer_kwargs)
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
            state = unpad_initial_states[i].clone().to(self.lbfgs_device)
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
                batch=batch,
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

    def act(self):
        return self.loop.run_until_complete(self._act_async())

    def select_action(self):
        output = self.act()

        return {key: value.cpu().numpy() for key, value in output.items()}
