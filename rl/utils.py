import torch
import numpy as np

from math import floor
from schnetpack import Properties
from schnetpack.data.loader import _collate_aseatoms
from schnetpack.nn.neighbors import atom_distances

from rl import DEVICE
from rl.replay_buffer import UNWANTED_KEYS


class TimelimitScheduler():
    def __init__(self,  timelimit_init=1, step=10, interval=100000, constant=True):
        self.init_tl = timelimit_init
        self.step = step
        self.interval = interval
        self.constant = constant

    def update(self, current_step):
        if not self.constant:
            self.tl = self.init_tl + self.step * (current_step // self.interval)
        else:
            self.tl = self.init_tl

    def get_timelimit(self):
        return self.tl


def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss

def recollate_batch(state_batch, indices, new_state_batch):
    # Transform state_batch and new_state_batch to lists.
    bs = state_batch['_positions'].shape[0]
    states = [{k:v[i].cpu() for k, v in state_batch.items() if k not in UNWANTED_KEYS} for i in range(bs)]
    
    new_bs = new_state_batch['_positions'].shape[0]
    new_states = [{k:v[i].cpu() for k, v in new_state_batch.items() if k not in UNWANTED_KEYS} for i in range(new_bs)]
    
    # Replaces some states with new ones and collates them into batch.
    for i, ind in enumerate(indices):
        states[ind] = new_states[i]
    return {k:v.to(DEVICE) for k, v in _collate_aseatoms(states).items()}

def calculate_molecule_metrics(state, next_state, cutoff_network):
    n_atoms = state[Properties.atom_mask].sum(dim=1).long()
    n_mol = n_atoms.shape[0]

    # get interatomic vectors and distances for state
    rij = atom_distances(
        positions=state[Properties.R],
        neighbors=state[Properties.neighbors],
        neighbor_mask=state[Properties.neighbor_mask],
        cell=state[Properties.cell],
        cell_offsets=state[Properties.cell_offset]
    )

    min_r, avg_r, max_r = 0, 0, 0
    for r, n in zip(rij, n_atoms):
        current_molecule_r = r[:n, :n - 1]
        min_r += current_molecule_r.min()
        avg_r += current_molecule_r.mean()
        max_r += current_molecule_r.max()
    min_r, avg_r, max_r = min_r / n_mol, avg_r / n_mol, max_r / n_mol

    fcut = cutoff_network(rij) * state[Properties.neighbor_mask]
    avg_atoms_in_cutoff_before = fcut.sum(dim=(1, 2)) / n_atoms

    n_atoms_ns = next_state[Properties.atom_mask].sum(dim=1).long()
    # get interatomic vectors and distances for next state
    rij_ns = atom_distances(
        positions=next_state[Properties.R],
        neighbors=next_state[Properties.neighbors],
        neighbor_mask=next_state[Properties.neighbor_mask],
        cell=next_state[Properties.cell],
        cell_offsets=next_state[Properties.cell_offset]
    )
    fcut_ns = cutoff_network(rij_ns) * next_state[Properties.neighbor_mask]
    avg_atoms_in_cutoff_after = fcut_ns.sum(dim=(1, 2)) / n_atoms_ns


    metrics = {
        "Molecule/min_interatomic_dist": min_r.item(),
        "Molecule/avg_interatomic_dist": avg_r.item(),
        "Molecule/max_interatomic_dist": max_r.item(),
        "Molecule/avg_atoms_inside_cutoff_state": avg_atoms_in_cutoff_before.mean().item(),
        "Molecule/avg_atoms_inside_cutoff_next_state": avg_atoms_in_cutoff_after.mean().item()
    }
    return metrics

def calculate_gradient_norm(model):
    total_norm = 0.0
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in params:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    return total_norm

def calculate_action_norm(actions, atom_mask):
    num_atoms = atom_mask.sum(-1).long()
    actions_list = [action[:num_atoms[i]] for i, action in enumerate(actions)]
    mean_norm = np.array([np.linalg.norm(action, axis=1).mean() for action in actions_list]).mean()
    return mean_norm
