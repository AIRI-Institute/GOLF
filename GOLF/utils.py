import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter

from GOLF import DEVICE
from GOLF.optim.lion_pytorch import Lion


class LRCosineAnnealing:
    def __init__(self, lr, lr_min=1e-5, t_max=1000):
        self.lr = lr
        self.lr_min = lr_min
        self.t_max = t_max

    def get(self, t):
        return torch.FloatTensor(
            [
                self.lr_min
                + 0.5
                * (self.lr - self.lr_min)
                * (1 + np.cos(min(t_, self.t_max) * np.pi / self.t_max))
                for t_ in t
            ]
        ).to(DEVICE)


class LRConstant:
    def __init__(self, lr):
        self.lr = lr

    def get(self, t):
        return torch.FloatTensor([self.lr for t_ in t]).to(DEVICE)


def get_conformation_lr_scheduler(lr_scheduler_type, lr, t_max):
    if lr_scheduler_type == "Constant":
        return LRConstant(lr)
    elif lr_scheduler_type == "CosineAnnealing":
        return LRCosineAnnealing(lr, t_max=t_max)
    else:
        raise ValueError(
            "Unknown conformation LR scheduler type: {}".format(lr_scheduler_type)
        )


def get_lr_scheduler(scheduler_type, optimizer, **kwargs):
    if scheduler_type == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=10 * kwargs["initial_lr"],
            final_div_factor=kwargs["final_div_factor"],
            total_steps=kwargs["total_steps"],
            last_epoch=kwargs["last_epoch"],
        )
    elif scheduler_type == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs["total_steps"],
            eta_min=kwargs["initial_lr"] / kwargs["final_div_factor"],
            last_epoch=kwargs["last_epoch"],
        )
    elif scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs["total_steps"] // 3, gamma=kwargs["gamma"]
        )
    else:
        raise ValueError("Unknown LR scheduler type: {}".format(scheduler_type))


def get_optimizer_class(optimizer_name):
    if optimizer_name == "adam":
        return torch.optim.Adam
    elif optimizer_name == "lion":
        return Lion
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def recollate_batch(batch, indices, new_batch):
    # Transform state_batch and new_state_batch to lists.
    individual_states = batch.to_data_list()
    new_individual_states = new_batch.to_data_list()

    # Replaces some states with new ones and collates them into batch.
    for new_idx, idx in enumerate(indices):
        individual_states[idx] = new_individual_states[new_idx]
    return Batch.from_data_list(individual_states).to(DEVICE)


def calculate_atoms_in_cutoff(batch, idx_i):
    atoms_indices_range = get_atoms_indices_range(batch)
    # TODO check dimensions for edge indices
    indices, counts = torch.unique(idx_i, sorted=False, return_counts=True)
    n_atoms_expanded = torch.ones_like(indices)
    for molecule_id in range(batch.batch_size):
        molecule_indices = (indices >= atoms_indices_range[molecule_id]) & (
            indices < atoms_indices_range[molecule_id + 1]
        )
        n_atoms_expanded[molecule_indices] = atoms_indices_range[molecule_id + 1]

    return (counts / n_atoms_expanded).sum() / batch.batch_size


def calculate_molecule_metrics(batch, next_batch, cutoff=5.0, max_num_neighbors=32):
    n_atoms = get_n_atoms(batch)
    atoms_indices_range = get_atoms_indices_range(batch)

    idx_i, idx_j = radius_graph(
        batch.pos, r=cutoff, batch=batch.batch, max_num_neighbors=max_num_neighbors
    )
    next_idx_i, _ = radius_graph(
        next_batch.pos,
        r=cutoff,
        batch=next_batch.batch,
        max_num_neighbors=max_num_neighbors,
    )

    assert (
        batch.batch.size(0) == atoms_indices_range[-1].item()
    ), "Assume that all atoms are listed in _idx_m property!"

    min_r, avg_r, max_r = 0, 0, 0
    rij = (batch.pos[idx_i] - batch.pos[idx_j]).pow(2).sum(dim=-1).sqrt()

    for molecule_id in range(n_atoms.size(0)):
        molecule_indices = (idx_i >= atoms_indices_range[molecule_id]) & (
            idx_j < atoms_indices_range[molecule_id + 1]
        )
        current_molecule_r = rij[molecule_indices]
        min_r += current_molecule_r.min()
        avg_r += current_molecule_r.mean()
        max_r += current_molecule_r.max()

    min_r, avg_r, max_r = (
        min_r / batch.batch_size,
        avg_r / batch.batch_size,
        max_r / batch.batch_size,
    )
    avg_atoms_in_cutoff_before = calculate_atoms_in_cutoff(batch, idx_i)
    avg_atoms_in_cutoff_after = calculate_atoms_in_cutoff(next_batch, next_idx_i)

    metrics = {
        "Molecule/min_interatomic_dist": min_r.item(),
        "Molecule/avg_interatomic_dist": avg_r.item(),
        "Molecule/max_interatomic_dist": max_r.item(),
        "Molecule/avg_atoms_inside_cutoff_state": avg_atoms_in_cutoff_before.item(),
        "Molecule/avg_atoms_inside_cutoff_next_state": avg_atoms_in_cutoff_after.item(),
    }

    return metrics


def calculate_gradient_norm(model):
    total_norm = 0.0
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in params:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm**2
    total_norm = total_norm ** (0.5)
    return total_norm


def calculate_action_norm(actions, cumsum_numbers_atoms):
    actions_norm = np.linalg.norm(actions, axis=1)
    mean_norm = 0
    for idx in range(len(cumsum_numbers_atoms) - 1):
        mean_norm += actions_norm[
            cumsum_numbers_atoms[idx] : cumsum_numbers_atoms[idx + 1]
        ].mean()

    return mean_norm / (len(cumsum_numbers_atoms) - 1)


def get_atoms_indices_range(batch):
    n_atoms = get_n_atoms(batch)
    return torch.nn.functional.pad(torch.cumsum(n_atoms, dim=0), pad=(1, 0))


def get_n_atoms(batch):
    return scatter(torch.ones_like(batch.batch), batch.batch, dim_size=batch.batch_size)
