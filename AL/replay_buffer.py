import torch
import numpy as np

from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.nn import scatter_add

from AL.utils import unpad_state


class ReplayBufferGD(object):
    def __init__(self, device, max_size=int(1e6), atomrefs=None):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = [None] * self.max_size
        self.energy = torch.empty((max_size, 1), dtype=torch.float32)
        self.forces = [None] * self.max_size

        if atomrefs:
            self.atomrefs = torch.tensor(atomrefs, device=device)
        else:
            self.atomrefs = None

    def add(self, states, forces, energies):
        energies = torch.tensor(energies, dtype=torch.float32)
        individual_states = unpad_state(states)
        # Update replay buffer
        for i in range(len(energies)):
            self.states[self.ptr] = individual_states[i]
            self.energy[self.ptr] = energies[i]
            self.forces[self.ptr] = torch.from_numpy(forces[i])
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        states = [self.states[i] for i in ind]
        state_batch = {
            key: value.to(self.device)
            for key, value in _atoms_collate_fn(states).items()
        }
        forces = torch.cat([self.forces[i] for i in ind]).to(self.device)

        energy = self.energy[ind].to(self.device)
        if self.atomrefs is not None:
            # Get system index
            idx_m = state_batch[properties.idx_m]

            # Get num molecules in the batch
            max_m = int(idx_m[-1]) + 1

            # Get atomization energy for each molecule in the batch
            atomization_energy = scatter_add(
                self.atomrefs[state_batch[properties.Z]], idx_m, dim_size=max_m
            ).unsqueeze(-1)
            energy -= atomization_energy

        return state_batch, forces, energy
