import asyncio
import collections
import concurrent.futures
from dataclasses import dataclass
import multiprocessing
import queue
from functools import partial

import schnetpack as spk
import torch
import torch.nn as nn
from schnetpack import properties
from torch.linalg import vector_norm
from torch.optim import LBFGS

from AL import DEVICE
from AL.utils import (
    get_action_scale_scheduler,
    get_atoms_indices_range,
    unpad_state,
    _atoms_collate_fn,
)
from utils.utils import ignore_extra_args
from AL.optim import lbfgs

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
        lbfgs_device="cuda",
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
        self.lbfgs_device = torch.device(lbfgs_device)

    def reset(self, initial_states, indices=None):
        if indices is None:
            indices = torch.arange(self.n_parallel)
        unpad_initial_states = unpad_state(initial_states)
        for i, idx in enumerate(indices):
            self.states[idx] = {
                k: v.detach().clone().to(self.lbfgs_device)
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
        energy = torch.zeros(self.n_parallel, device=self.lbfgs_device)
        n_iter = [0] * self.n_parallel

        # Define reevaluation function
        def closure(idx):
            self.optimizer_list[idx].zero_grad()
            # train=True to get correct gradients
            state = {key: value.to(DEVICE) for key, value in self.states[idx].items()}
            output = self.actor(state, t, train=True)
            self.states[idx][properties.R].grad = (
                -output["anti_gradient"].detach().to(self.lbfgs_device)
            )
            energy[idx] = output["energy"].to(self.lbfgs_device)
            n_iter[idx] += 1
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
            "n_iter": n_iter,
        }

    def select_action(self, t, return_n_iter=False):
        output = self.act(t)
        action = output["action"].cpu().numpy()
        energy = output["energy"].detach().cpu().numpy()
        done = output["done"].cpu().numpy()
        if return_n_iter:
            return action, energy, done, output["n_iter"]

        return action, energy, done


class ConformationOptimizer:
    def __init__(
        self,
        state: dict,
        policy2optimizer_queue: queue.Queue,
        optimizer2policy_queue: queue.Queue,
        optimizer_kwargs: dict,
        grad_threshold: float,
    ):
        super().__init__()
        self.state = state
        self.state[properties.R].requires_grad_(True)
        self.policy2optimizer_queue = policy2optimizer_queue
        self.optimizer2policy_queue = optimizer2policy_queue
        self.optimizer = LBFGS([self.state[properties.R]], **optimizer_kwargs)
        self.energy = None
        self.n_iter = None
        self.grad_threshold = grad_threshold

    def closure(self):
        self.optimizer.zero_grad()
        self.optimizer2policy_queue.put(self.state)
        anti_gradient, energy = self.policy2optimizer_queue.get()
        self.state[properties.R].grad = -anti_gradient
        self.energy = energy
        self.n_iter += 1
        return energy

    def step(self):
        self.n_iter = 0
        previous_position = self.state[properties.R].detach().clone()
        self.optimizer.step(self.closure)
        self.optimizer2policy_queue.put(None)
        done = torch.unsqueeze(
            self.optimizer._gather_flat_grad().abs().max() <= self.grad_threshold, dim=0
        )
        action = self.state[properties.R].detach().clone() - previous_position
        is_finite_action = torch.isfinite(action).all().unsqueeze(dim=0)
        return {
            "action": action,
            "energy": self.energy,
            "done": done,
            "is_finite_action": is_finite_action,
            "n_iter": torch.tensor([self.n_iter]),
        }


class ALMultiThreadingPolicy(nn.Module):
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
        lbfgs_device="cuda",
    ):
        super().__init__()
        self.n_parallel = n_parallel
        self.action_scale = action_scale
        self.max_iter = max_iter
        self.grad_threshold = grad_threshold
        self.optimizer_kwargs = {"lr": self.action_scale, "max_iter": self.max_iter}
        self.actor = Actor(
            backbone,
            backbone_args,
            action_scale,
            action_scale_scheduler,
            action_norm_limit,
        )
        self.lbfgs_device = torch.device(lbfgs_device)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel)
        self.policy2optimizer_queues = [
            queue.Queue(maxsize=1) for _ in range(n_parallel)
        ]
        self.optimizer2policy_queues = [
            queue.Queue(maxsize=1) for _ in range(n_parallel)
        ]
        self.conformation_optimizers = [None] * self.n_parallel

    def reset(self, initial_states, indices=None):
        if indices is None:
            indices = torch.arange(self.n_parallel)
        unpad_initial_states = unpad_state(initial_states)
        for i, idx in enumerate(indices):
            state = {
                key: value.detach().clone().to(self.lbfgs_device)
                for key, value in unpad_initial_states[i].items()
            }
            self.conformation_optimizers[idx] = ConformationOptimizer(
                state,
                self.policy2optimizer_queues[idx],
                self.optimizer2policy_queues[idx],
                self.optimizer_kwargs,
                self.grad_threshold,
            )

    def act(self, t):
        futures = []
        for conformation_optimizer in self.conformation_optimizers:
            futures.append(self.executor.submit(conformation_optimizer.step))

        conformation_optimizers_ids = set(range(self.n_parallel))
        while True:
            individual_states = {}
            stopped_optimizers_ids = set()
            for conformation_optimizer_id in conformation_optimizers_ids:
                individual_state = self.optimizer2policy_queues[
                    conformation_optimizer_id
                ].get()
                if individual_state is None:
                    stopped_optimizers_ids.add(conformation_optimizer_id)
                    continue

                individual_states[conformation_optimizer_id] = individual_state

            conformation_optimizers_ids -= stopped_optimizers_ids
            if len(individual_states) == 0:
                break

            states = _atoms_collate_fn(list(individual_states.values()))
            states = {key: value.to(DEVICE) for key, value in states.items()}
            output = self.actor(states, t, train=True)
            anti_gradients = torch.split(
                output["anti_gradient"].detach().to(self.lbfgs_device),
                states[properties.n_atoms].tolist(),
            )
            energies = output["energy"].detach().to(self.lbfgs_device).view(-1, 1)
            for i, optimizer_idx in enumerate(individual_states.keys()):
                self.policy2optimizer_queues[optimizer_idx].put(
                    (anti_gradients[i], energies[i])
                )

        result = collections.defaultdict(list)
        for future in futures:
            for key, value in future.result().items():
                result[key].append(value)

        for key, value in result.items():
            result[key] = torch.cat(value, dim=0)

        return result

    def select_action(self, t, return_n_iter=False):
        output = self.act(t)
        action = output["action"].cpu().numpy()
        energy = output["energy"].detach().cpu().numpy()
        done = output["done"].cpu().numpy()
        is_finite_action = output["is_finite_action"].cpu().numpy()
        n_iter = output["n_iter"].cpu().numpy()
        if return_n_iter:
            return action, energy, done, is_finite_action, n_iter

        return action, energy, done


class AsyncConformationOptimizer:
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
        self.state[properties.R].requires_grad_(True)
        self.policy2optimizer_queue = policy2optimizer_queue
        self.optimizer2policy_queue = optimizer2policy_queue
        self.optimizer = lbfgs.LBFGS([self.state[properties.R]], **optimizer_kwargs)
        self.energy = None
        self.n_iter = None
        self.grad_threshold = grad_threshold

    async def closure(self):
        self.optimizer.zero_grad()
        await self.optimizer2policy_queue.put(self.state)
        anti_gradient, energy = await self.policy2optimizer_queue.get()
        self.state[properties.R].grad = -anti_gradient
        self.energy = energy
        self.n_iter += 1
        return energy

    async def step(self):
        self.n_iter = 0
        previous_position = self.state[properties.R].detach().clone()
        await self.optimizer.step(self.closure)
        await self.optimizer2policy_queue.put(None)
        done = torch.unsqueeze(
            self.optimizer._gather_flat_grad().abs().max() <= self.grad_threshold, dim=0
        )
        action = self.state[properties.R].detach().clone() - previous_position
        is_finite_action = torch.isfinite(action).all().unsqueeze(dim=0)
        return {
            "action": action,
            "energy": self.energy,
            "done": done,
            "is_finite_action": is_finite_action,
            "n_iter": torch.tensor([self.n_iter]),
        }


class ALAsyncPolicy(nn.Module):
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
        lbfgs_device="cuda",
    ):
        super().__init__()
        self.n_parallel = n_parallel
        self.action_scale = action_scale
        self.max_iter = max_iter
        self.grad_threshold = grad_threshold
        self.optimizer_kwargs = {"lr": self.action_scale, "max_iter": self.max_iter}
        self.actor = Actor(
            backbone,
            backbone_args,
            action_scale,
            action_scale_scheduler,
            action_norm_limit,
        )
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
        unpad_initial_states = unpad_state(initial_states)
        torch.set_grad_enabled(True)
        for i, idx in enumerate(indices):
            state = {
                key: value.detach().clone().to(self.lbfgs_device)
                for key, value in unpad_initial_states[i].items()
            }
            self.conformation_optimizers[idx] = AsyncConformationOptimizer(
                state,
                self.policy2optimizer_queues[idx],
                self.optimizer2policy_queues[idx],
                self.optimizer_kwargs,
                self.grad_threshold,
            )

    async def _act_task(self, t):
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

            states = _atoms_collate_fn(list(individual_states.values()))
            torch.set_grad_enabled(True)
            states = {key: value.to(DEVICE) for key, value in states.items()}
            output = self.actor(states, t, train=True)
            anti_gradients = torch.split(
                output["anti_gradient"].detach().to(self.lbfgs_device),
                states[properties.n_atoms].tolist(),
            )
            energies = output["energy"].detach().to(self.lbfgs_device).view(-1, 1)
            for i, optimizer_idx in enumerate(individual_states.keys()):
                await self.policy2optimizer_queues[optimizer_idx].put(
                    (anti_gradients[i], energies[i])
                )

    async def _act_async(self, t):
        tasks = [
            conformation_optimizer.step()
            for conformation_optimizer in self.conformation_optimizers
        ]
        tasks.append(self._act_task(t))
        task_results = await asyncio.gather(*tasks)

        result = collections.defaultdict(list)
        for task in task_results[:-1]:
            for key, value in task.items():
                result[key].append(value)

        for key, value in result.items():
            result[key] = torch.cat(value, dim=0)

        return result

    def act(self, t):
        return self.loop.run_until_complete(self._act_async(t))

    def select_action(self, t, return_n_iter=False):
        output = self.act(t)
        action = output["action"].cpu().numpy()
        energy = output["energy"].detach().cpu().numpy()
        done = output["done"].cpu().numpy()
        is_finite_action = output["is_finite_action"].cpu().numpy()
        n_iter = output["n_iter"].cpu().numpy()
        if return_n_iter:
            return action, energy, done, is_finite_action, n_iter

        return action, energy, done


@dataclass
class MethodCallMessage:
    method: str
    kwargs: dict
    name: str = "method_call_message"


@dataclass
class StepResultMessage:
    action: torch.Tensor
    energy: torch.Tensor
    done: torch.Tensor
    is_finite_action: torch.Tensor
    n_iter: torch.Tensor
    name: str = "step_result_message"


@dataclass
class CurrentStateMessage:
    state: dict
    name: str = "current_state_message"


@dataclass
class ModelOutputMessage:
    anti_gradient: torch.Tensor
    energy: torch.Tensor
    name: str = "model_output_message"


class ConformationOptimizerProcess(multiprocessing.Process):
    def __init__(
        self, read_queue, write_queue, optimizer_kwargs, grad_threshold, device
    ):
        super().__init__()
        self.optimizer_kwargs = optimizer_kwargs
        self.device = torch.device(device)
        self.cpu = torch.device("cpu")
        self.read_queue = read_queue
        self.write_queue = write_queue
        self.grad_threshold = grad_threshold
        self.state = None
        self.optimizer = None
        self.energy = None
        self.n_iter = None

    def set_state(self, state):
        self.state = {key: value.to(self.device) for key, value in state.items()}
        self.state[properties.R].requires_grad_(True)
        self.optimizer = LBFGS([self.state[properties.R]], **self.optimizer_kwargs)
        self.energy = None
        self.n_iter = None

    def run(self):
        while True:
            method_call_message = self.read_queue.recv()
            if method_call_message is None:
                break

            getattr(self, method_call_message.method)(**method_call_message.kwargs)

    def closure(self):
        self.optimizer.zero_grad()
        self.write_queue.send(
            CurrentStateMessage(
                state={
                    key: value.detach().to(self.cpu)
                    for key, value in self.state.items()
                }
            )
        )
        model_output_message = self.read_queue.recv()
        self.state[properties.R].grad = -model_output_message.anti_gradient.to(
            self.device
        )
        self.energy = model_output_message.energy.to(self.device)
        self.n_iter += 1
        return self.energy

    def step(self):
        self.n_iter = 0
        previous_position = self.state[properties.R].detach().clone()
        self.optimizer.step(self.closure)
        done = torch.unsqueeze(
            self.optimizer._gather_flat_grad().abs().max() <= self.grad_threshold, dim=0
        )
        action = self.state[properties.R].detach().clone() - previous_position
        is_finite_action = torch.isfinite(action).all().unsqueeze(dim=0)
        self.write_queue.send(
            StepResultMessage(
                action=action.detach().to(self.cpu),
                energy=self.energy.detach().to(self.cpu),
                done=done.detach().to(self.cpu),
                is_finite_action=is_finite_action.detach().to(self.cpu),
                n_iter=torch.tensor([self.n_iter]),
            )
        )


class ALMultiProcessingPolicy(nn.Module):
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
        lbfgs_device="cuda",
    ):
        super().__init__()
        self.n_parallel = n_parallel
        self.action_scale = action_scale
        self.max_iter = max_iter
        self.grad_threshold = grad_threshold
        self.optimizer_kwargs = {"lr": self.action_scale, "max_iter": self.max_iter}
        self.lbfgs_device = torch.device(lbfgs_device)
        self.cpu = torch.device("cpu")
        self.actor = Actor(
            backbone,
            backbone_args,
            action_scale,
            action_scale_scheduler,
            action_norm_limit,
        )

        multiprocessing.set_start_method("spawn", force=True)

        self.policy2optimizer_queues = []
        self.optimizer2policy_queues = []
        self.conformation_optimizers = []
        for _ in range(self.n_parallel):
            policy2optimizer1, policy2optimizer2 = multiprocessing.Pipe(duplex=False)
            optimizer2policy1, optimizer2policy2 = multiprocessing.Pipe(duplex=False)
            self.policy2optimizer_queues.append(policy2optimizer2)
            self.optimizer2policy_queues.append(optimizer2policy1)
            optimizer_process = ConformationOptimizerProcess(
                policy2optimizer1,
                optimizer2policy2,
                self.optimizer_kwargs,
                self.grad_threshold,
                self.lbfgs_device,
            )
            self.conformation_optimizers.append(optimizer_process)
            optimizer_process.start()

    def reset(self, initial_states, indices=None):
        if indices is None:
            indices = torch.arange(self.n_parallel)
        unpad_initial_states = unpad_state(initial_states)
        for i, idx in enumerate(indices):
            state = {
                key: value.detach().to(self.cpu)
                for key, value in unpad_initial_states[i].items()
            }
            self.policy2optimizer_queues[idx].send(
                MethodCallMessage("set_state", {"state": state})
            )

    def act(self, t):
        for policy2optimizer_queue in self.policy2optimizer_queues:
            policy2optimizer_queue.send(MethodCallMessage("step", {}))

        conformation_optimizers_ids = set(range(self.n_parallel))
        step_result_messages = [None] * self.n_parallel
        while True:
            individual_states = {}
            stopped_optimizers_ids = set()
            for conformation_optimizer_id in conformation_optimizers_ids:
                message = self.optimizer2policy_queues[conformation_optimizer_id].recv()
                if message.name == "current_state_message":
                    individual_state = message.state
                    individual_states[conformation_optimizer_id] = individual_state
                elif message.name == "step_result_message":
                    stopped_optimizers_ids.add(conformation_optimizer_id)
                    step_result_messages[conformation_optimizer_id] = message
                    continue

            conformation_optimizers_ids -= stopped_optimizers_ids
            if len(individual_states) == 0:
                break

            states = _atoms_collate_fn(list(individual_states.values()))
            states = {key: value.to(DEVICE) for key, value in states.items()}
            output = self.actor(states, t, train=True)
            anti_gradients = torch.split(
                output["anti_gradient"].detach(), states[properties.n_atoms].tolist()
            )
            energies = output["energy"].detach().view(-1, 1)
            for i, optimizer_idx in enumerate(individual_states.keys()):
                self.policy2optimizer_queues[optimizer_idx].send(
                    ModelOutputMessage(
                        anti_gradients[i].to(self.cpu), energies[i].to(self.cpu)
                    )
                )

        result_actions = []
        result_energies = []
        result_dones = []
        result_is_finite_actions = []
        result_n_iters = []

        for step_result_message in step_result_messages:
            result_actions.append(step_result_message.action)
            result_energies.append(step_result_message.energy)
            result_dones.append(step_result_message.done)
            result_is_finite_actions.append(step_result_message.is_finite_action)
            result_n_iters.append(step_result_message.n_iter)

        result = {
            "action": torch.cat(result_actions, dim=0),
            "energy": torch.cat(result_energies, dim=0),
            "done": torch.cat(result_dones, dim=0),
            "is_finite_action": torch.cat(result_is_finite_actions, dim=0),
            "n_iter": torch.cat(result_n_iters, dim=0),
        }

        return result

    def select_action(self, t, return_n_iter=False):
        output = self.act(t)
        action = output["action"].cpu().numpy()
        energy = output["energy"].detach().cpu().numpy()
        done = output["done"].cpu().numpy()
        is_finite_action = output["is_finite_action"].cpu().numpy()
        n_iter = output["n_iter"].cpu().numpy()
        if return_n_iter:
            return action, energy, done, is_finite_action, n_iter

        return action, energy, done

    def close(self):
        for policy2optimizer_queue in self.policy2optimizer_queues:
            policy2optimizer_queue.send(None)

        for process in self.conformation_optimizers:
            process.join()
