import torch
import numpy as np

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from schnetpack.data.loader import _collate_aseatoms


UNWANTED_KEYS = ["representation", "vector_representation"]

class ReplayBufferGD(object):
    def __init__(self, device, max_size=int(1e6)):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = [None] * self.max_size
        self.energy = torch.empty((max_size, 1), dtype=torch.float32)
        self.forces = [None] * self.max_size

    def add(self, states, forces, energies):
        energies = torch.tensor(energies, dtype=torch.float32)
        # Update replay buffer
        for i in range(len(energies)):
            self.states[self.ptr] = {k:v[i].cpu() for k, v in states.items() if k not in UNWANTED_KEYS}
            self.energy[self.ptr] = energies[i]
            self.forces[self.ptr] = torch.from_numpy(forces[i])
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, batch_size, replace=False)
        states = [self.states[i] for i in ind]
        state_batch = {key: value.to(self.device) for key, value in _collate_aseatoms(states).items()}
        forces = _collate_forces(
            [self.forces[i] for i in ind],
            max_size=state_batch['_positions'].size(1)
        ).to(self.device)
        energy = self.energy[ind].to(self.device)
        return state_batch, forces, energy


def _collate_forces(forces, max_size=None):
    if max_size is None:
        max_size = max([force.shape[0] for force in forces])
    forces_batch = torch.zeros(len(forces), max_size, forces[0].shape[1])
    for i, force in enumerate(forces):
        forces_batch[i, slice(0, force.shape[0])] = force
    return forces_batch
