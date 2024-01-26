import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from env.optimization_env import OptimizationEnv

NORM_THRESHOLD = 0.04


class ReplayBuffer(object):
    def __init__(
        self,
        device,
        max_size,
        max_total_conformations,
        atomrefs=None,
        initial_RB=None,
        eval_RB=None,
        initial_conf_pct=0.0,
        filter_forces_by_norm=True,
    ):
        self.device = device
        self.max_size = max_size
        self.max_total_conformations = max_total_conformations
        self.filter_forces_by_norm = filter_forces_by_norm
        self.initial_RB = initial_RB
        self.eval_RB = eval_RB
        self.ptr = 0
        self.size = 0
        self.replay_buffer_full = False

        if self.initial_RB:
            self.initial_conf_pct = initial_conf_pct
        else:
            self.initial_conf_pct = 0.0

        self.states = [None] * self.max_size
        self.energy = torch.empty((max_size, 1), dtype=torch.float32)
        self.forces = [None] * self.max_size

        if atomrefs:
            self.atomrefs = torch.tensor(atomrefs, device=device)
        else:
            self.atomrefs = None

    def add(self, batch, forces, energies):
        energies = torch.tensor(energies, dtype=torch.float32)
        force_norms = np.array([np.linalg.norm(force) for force in forces])
        individual_states = batch.to_data_list()
        # Exclude conformations with forces that have a high norm
        # from the replay buffer
        if self.filter_forces_by_norm:
            indices = np.where(force_norms < NORM_THRESHOLD)[0]
        else:
            indices = np.arange(len(individual_states))
        for i in indices:
            self.states[self.ptr] = individual_states[i]
            self.energy[self.ptr] = energies[i]
            self.forces[self.ptr] = torch.tensor(forces[i], dtype=torch.float32)
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = self.size + 1

        self.replay_buffer_full = self.size >= self.max_total_conformations

    def sample(self, batch_size):
        new_samples_batch_size = int(batch_size * (1 - self.initial_conf_pct))
        states, forces, energy = self.sample_wo_collate(new_samples_batch_size)

        if self.initial_RB and self.initial_conf_pct:
            initial_conf_batch_size = batch_size - new_samples_batch_size
            init_states, init_forces, init_energy = self.initial_RB.sample_wo_collate(
                initial_conf_batch_size
            )
            states = states + init_states
            forces = forces + init_forces
            energy = torch.cat((energy, init_energy), dim=0)

        batch = Batch.from_data_list(states).to(self.device)
        forces = torch.cat(forces).to(self.device)

        if self.atomrefs is not None:
            # Get atomization energy for each molecule in the batch
            atomization_energy = scatter(
                self.atomrefs[batch.z],
                batch.batch,
                dim_size=batch.batch_size,
            ).unsqueeze(-1)
            energy -= atomization_energy

        return batch, forces, energy

    def sample_eval(self, batch_size):
        states, forces, energy = self.sample_wo_collate(batch_size)
        batch = Batch.from_data_list(states).to(self.device)
        forces = torch.cat(forces).to(self.device)
        energy = energy.to(self.device)
        if self.atomrefs is not None:
            # Get atomization energy for each molecule in the batch
            atomization_energy = scatter(
                self.atomrefs[batch.z],
                batch.batch,
                dim_size=batch.batch_size,
            ).unsqueeze(-1)
            energy -= atomization_energy
        return batch, forces, energy

    def sample_wo_collate(self, batch_size):
        ind = np.random.choice(min(self.size, self.max_size), batch_size, replace=False)
        states = [self.states[i] for i in ind]
        forces = [self.forces[i] for i in ind]
        energy = self.energy[ind].to(self.device)
        return states, forces, energy


def fill_initial_replay_buffer(device, db_path, args, atomrefs=None):
    # Env kwargs
    env_kwargs = {
        "db_path": db_path,
        "n_parallel": 1,
        "timelimit": args.timelimit_train,
        "sample_initial_conformations": False,
        "num_initial_conformations": args.num_initial_conformations,
    }
    # Initialize env
    env = OptimizationEnv(**env_kwargs)
    if args.num_initial_conformations == -1:
        total_confs = env.get_db_length()
    else:
        total_confs = args.num_initial_conformations

    initial_replay_buffer = ReplayBuffer(
        device,
        max_size=total_confs,
        max_total_conformations=total_confs,
        atomrefs=atomrefs,
        filter_forces_by_norm=False,
    )

    # Fill up the replay buffer
    for _ in range(total_confs):
        state = env.reset()
        # Save initial state in replay buffer
        energies = np.array([env.energy])
        forces = [np.array(force) for force in env.force]
        initial_replay_buffer.add(state, forces, energies)

    return initial_replay_buffer
