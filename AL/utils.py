import numpy as np
import schnetpack.nn as snn
import torch
from schnetpack import properties

from AL import DEVICE
from AL.optim.lion_pytorch import Lion


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
        )
    elif scheduler_type == "CosineAnnealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs["total_steps"],
            eta_min=kwargs["initial_lr"] / kwargs["final_div_factor"],
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


def recollate_batch(state_batch, indices, new_state_batch):
    # Transform state_batch and new_state_batch to lists.
    individual_states = unpad_state(state_batch)
    new_individual_states = unpad_state(new_state_batch)

    # Replaces some states with new ones and collates them into batch.
    for new_idx, idx in enumerate(indices):
        individual_states[idx] = new_individual_states[new_idx]
    return {k: v.to(DEVICE) for k, v in _atoms_collate_fn(individual_states).items()}


def calculate_atoms_in_cutoff(state):
    n_atoms = state[properties.n_atoms]
    atoms_indices_range = get_atoms_indices_range(state)
    indices, counts = torch.unique(
        state[properties.idx_i], sorted=False, return_counts=True
    )
    n_atoms_expanded = torch.ones_like(indices)
    for molecule_id in range(n_atoms.size(0)):
        molecule_indices = (atoms_indices_range[molecule_id] <= indices) & (
            indices < atoms_indices_range[molecule_id + 1]
        )
        n_atoms_expanded[molecule_indices] = n_atoms[molecule_id]

    return torch.sum(counts / (n_atoms_expanded * n_atoms.size(0)))


def calculate_molecule_metrics(state, next_state):
    n_atoms = state[properties.n_atoms]
    atoms_indices_range = get_atoms_indices_range(state)
    assert (
        state[properties.idx_m].size(0) == atoms_indices_range[-1].item()
    ), "Assume that all atoms are listed in _idx_m property!"

    min_r, avg_r, max_r = 0, 0, 0
    rij = torch.linalg.norm(state[properties.Rij], dim=-1)
    for molecule_id in range(n_atoms.size(0)):
        molecule_indices = (
            atoms_indices_range[molecule_id] <= state[properties.idx_i]
        ) & (state[properties.idx_j] < atoms_indices_range[molecule_id + 1])
        current_molecule_r = rij[molecule_indices]
        min_r += current_molecule_r.min()
        avg_r += current_molecule_r.mean()
        max_r += current_molecule_r.max()

    min_r, avg_r, max_r = (
        min_r / n_atoms.size(0),
        avg_r / n_atoms.size(0),
        max_r / n_atoms.size(0),
    )
    avg_atoms_in_cutoff_before = calculate_atoms_in_cutoff(state)
    avg_atoms_in_cutoff_after = calculate_atoms_in_cutoff(next_state)

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


def get_cutoff_by_string(cutoff_type):
    if cutoff_type == "cosine":
        return snn.cutoff.CosineCutoff

    raise ValueError(f"Unexpected cutoff type:{cutoff_type}")


def get_atoms_indices_range(states):
    return torch.nn.functional.pad(
        torch.cumsum(states[properties.n_atoms], dim=0), pad=(1, 0)
    )


def unpad_state(states):
    individual_states = []
    n_molecules = states[properties.n_atoms].size(0)
    n_atoms = get_atoms_indices_range(states)
    for i in range(n_molecules):
        state = {
            properties.n_atoms: torch.unsqueeze(n_atoms[i + 1] - n_atoms[i], dim=0)
            .clone()
            .cpu()
        }
        for key in (properties.Z, properties.position):
            state[key] = states[key][n_atoms[i] : n_atoms[i + 1]].clone().cpu()

        for key in (properties.cell, properties.pbc, properties.idx):
            state[key] = states[key][i].unsqueeze(0).clone().cpu()

        assert (
            states[properties.idx_m].size(0) == n_atoms[-1].item()
        ), "Assume that all atoms are listed in _idx_m property!"
        molecule_indices = (n_atoms[i] <= states[properties.idx_i]) & (
            states[properties.idx_i] < n_atoms[i + 1]
        )
        for key in (properties.lidx_i, properties.lidx_j, properties.offsets):
            state[key] = states[key][molecule_indices].clone().cpu()

        for key in (properties.idx_i, properties.idx_j):
            state[key] = state[f"{key}_local"].clone().cpu()

        state[properties.idx_m] = torch.zeros_like(state[properties.Z])

        individual_states.append(state)

    return individual_states


def _atoms_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]
    idx_keys = {properties.idx_i, properties.idx_j, properties.idx_i_triples}
    # Atom triple indices must be treated separately
    idx_triple_keys = {properties.idx_j_triples, properties.idx_k_triples}

    coll_batch = {}
    for key in elem:
        if (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[properties.n_atoms], dim=0)
    seg_m = torch.cat(
        [torch.zeros((1,), dtype=seg_m.dtype, device=seg_m.device), seg_m], dim=0
    )
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch), device=seg_m.device),
        repeats=coll_batch[properties.n_atoms],
        dim=0,
    )
    coll_batch[properties.idx_m] = idx_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[properties.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch
